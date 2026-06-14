"""
Train first Doom Retro behavior-cloning model.

Input:
  vision_dataset_v2/doomretro_keyed_imitation/doomretro_easiest_level_completion_*/actions.csv

Output:
  checkpoints/doomretro_imitation_cnn.pt

This is version 1:
  single frame -> action

Later version:
  frame sequence -> action with GRU/LSTM
"""

from pathlib import Path
import random
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keyed_imitation"
OUT_PATH = ROOT / "checkpoints" / "doomretro_imitation_cnn.pt"

ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
    "no_op",
]

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def simplify_action(action: str):
    """
    Turn compound actions into one main action for version 1.

    This keeps the first trainer simple.
    Later we can train multi-label actions.
    """
    parts = set(str(action).split("+"))

    # Prioritize actions useful for level completion.
    if "use" in parts:
        return "use"
    if "shoot" in parts:
        return "shoot"

    if "move_forward" in parts:
        return "move_forward"
    if "strafe_left" in parts:
        return "strafe_left"
    if "strafe_right" in parts:
        return "strafe_right"
    if "turn_left" in parts:
        return "turn_left"
    if "turn_right" in parts:
        return "turn_right"
    if "move_backward" in parts:
        return "move_backward"

    return "no_op"


class DoomRetroImitationDataset(Dataset):
    def __init__(self):
        self.rows = []

        runs = sorted(DATA_ROOT.glob("doomretro_easiest_level_completion_*"))
        if not runs:
            raise SystemExit(f"No runs found in {DATA_ROOT}")

        print(f"[imitation] found runs: {len(runs)}")

        for run in runs:
            csv_path = run / "actions.csv"
            if not csv_path.exists():
                print(f"[imitation] skipping missing csv: {run}")
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                action = simplify_action(row["action"])

                # Downsample no_op so the model does not learn to stand still.
                if action == "no_op" and random.random() > 0.12:
                    continue

                # Mildly downsample move_forward from the huge first run.
                if action == "move_forward" and random.random() > 0.75:
                    continue

                frame_path = run / str(row["frame_path"])
                if not frame_path.exists():
                    continue

                if action not in ACTION_TO_IDX:
                    continue

                self.rows.append({
                    "frame_path": frame_path,
                    "action": action,
                    "label": ACTION_TO_IDX[action],
                    "run": run.name,
                })

        if len(self.rows) < 100:
            raise SystemExit("Not enough usable training rows.")

        counts = Counter(r["action"] for r in self.rows)
        print(f"[imitation] usable rows: {len(self.rows)}")
        print("[imitation] action counts:")
        for name, count in counts.most_common():
            print(f"  {name:<15} {count}")

        self.transform = T.Compose([
            T.Resize((84, 84)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img = Image.open(item["frame_path"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, y


class ImitationCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    random.seed(7)
    torch.manual_seed(7)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = DoomRetroImitationDataset()

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImitationCNN(num_actions=len(ACTION_NAMES)).to(device)

    labels = [r["label"] for r in dataset.rows]
    counts = torch.bincount(torch.tensor(labels), minlength=len(ACTION_NAMES)).float()

    # Weighted loss helps rare actions like use/shoot/backward.
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    weights = weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc = 0.0

    print(f"[imitation] device={device}")
    print("[imitation] class counts:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name:<15} {int(counts[i])}")

    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * len(y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += len(y)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = loss_fn(logits, y)

                val_loss_total += float(loss.item()) * len(y)
                pred = logits.argmax(dim=1)
                val_correct += int((pred == y).sum().item())
                val_total += len(y)

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[imitation] epoch={epoch:02d}/15 "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "action_names": ACTION_NAMES,
                "num_actions": len(ACTION_NAMES),
                "type": "doomretro_imitation_cnn_single_frame",
                "best_val_acc": best_val_acc,
                "data_root": str(DATA_ROOT),
            }, OUT_PATH)
            print(f"[imitation] saved {OUT_PATH}")

    print(f"[imitation] best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
