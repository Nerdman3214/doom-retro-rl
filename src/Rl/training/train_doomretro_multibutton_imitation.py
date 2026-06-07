"""
Train Doom Retro multi-button behavior cloning model.

Input:
  vision_dataset_v2/doomretro_keyed_imitation/doomretro_easiest_level_completion_*/actions.csv

Output:
  checkpoints/doomretro_imitation_multibutton_cnn.pt

This predicts 8 independent buttons:
  move_forward, move_backward, turn_left, turn_right,
  strafe_left, strafe_right, shoot, use
"""

from pathlib import Path
from collections import Counter
import random

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keyed_imitation"
OUT_PATH = ROOT / "checkpoints" / "doomretro_imitation_multibutton_cnn.pt"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

BUTTON_TO_IDX = {name: i for i, name in enumerate(BUTTON_NAMES)}


def action_to_multihot(action: str):
    parts = set(str(action).split("+"))
    y = torch.zeros(len(BUTTON_NAMES), dtype=torch.float32)

    for part in parts:
        if part in BUTTON_TO_IDX:
            y[BUTTON_TO_IDX[part]] = 1.0

    return y


class DoomRetroMultiButtonDataset(Dataset):
    def __init__(self):
        self.rows = []

        runs = sorted(DATA_ROOT.glob("doomretro_easiest_level_completion_*"))
        if not runs:
            raise SystemExit(f"No runs found in {DATA_ROOT}")

        print(f"[multi_bc] found runs: {len(runs)}")

        raw_counts = Counter()
        button_counts = Counter()
        kept_by_run = Counter()

        for run in runs:
            csv_path = run / "actions.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                action = str(row["action"])
                raw_counts[action] += 1

                # Downsample no_op so the model does not become passive.
                if action == "no_op" and random.random() > 0.10:
                    continue

                frame_path = run / str(row["frame_path"])
                if not frame_path.exists():
                    continue

                y = action_to_multihot(action)

                # Keep no_op as all zeros.
                for name, idx in BUTTON_TO_IDX.items():
                    if y[idx].item() > 0.5:
                        button_counts[name] += 1

                self.rows.append({
                    "frame_path": frame_path,
                    "action": action,
                    "target": y,
                    "run": run.name,
                })

                kept_by_run[run.name] += 1

        if len(self.rows) < 100:
            raise SystemExit("Not enough usable rows.")

        print(f"[multi_bc] usable rows: {len(self.rows)}")

        print("[multi_bc] kept rows by run:")
        for run, count in kept_by_run.items():
            print(f"  {run:<40} {count}")

        print("[multi_bc] top raw actions:")
        for action, count in raw_counts.most_common(20):
            print(f"  {action:<35} {count}")

        print("[multi_bc] button positives after filtering:")
        for name in BUTTON_NAMES:
            print(f"  {name:<15} {button_counts[name]}")

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
        y = item["target"]
        return x, y


class MultiButtonCNN(nn.Module):
    def __init__(self, num_buttons):
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
            nn.Linear(512, num_buttons),
        )

    def forward(self, x):
        return self.net(x)


def batch_metrics(logits, y, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    exact = (pred == y).all(dim=1).float().mean().item()

    # Button-level accuracy.
    button_acc = (pred == y).float().mean().item()

    # Positive precision/recall-ish diagnostics.
    true_pos = ((pred == 1) & (y == 1)).sum().item()
    pred_pos = (pred == 1).sum().item()
    actual_pos = (y == 1).sum().item()

    precision = true_pos / max(pred_pos, 1)
    recall = true_pos / max(actual_pos, 1)

    return exact, button_acc, precision, recall


def main():
    random.seed(7)
    torch.manual_seed(7)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = DoomRetroMultiButtonDataset()

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
    model = MultiButtonCNN(num_buttons=len(BUTTON_NAMES)).to(device)

    # Estimate per-button positive weights.
    all_targets = torch.stack([r["target"] for r in dataset.rows])
    positives = all_targets.sum(dim=0)
    negatives = len(dataset) - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)

    # Avoid insane weights for missing labels like turn/shoot.
    pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0).to(device)

    print(f"[multi_bc] device={device}")
    print("[multi_bc] pos_weight:")
    for i, name in enumerate(BUTTON_NAMES):
        print(f"  {name:<15} {float(pos_weight[i].detach().cpu()):.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_score = -1.0

    for epoch in range(1, 16):
        model.train()

        total_loss = 0.0
        total = 0

        train_exact_sum = 0.0
        train_button_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = len(y)
            total_loss += float(loss.item()) * n
            total += n

            exact, button_acc, _, _ = batch_metrics(logits.detach(), y)
            train_exact_sum += exact * n
            train_button_sum += button_acc * n

        train_loss = total_loss / max(total, 1)
        train_exact = train_exact_sum / max(total, 1)
        train_button_acc = train_button_sum / max(total, 1)

        model.eval()
        val_loss_total = 0.0
        val_total = 0

        val_exact_sum = 0.0
        val_button_sum = 0.0
        val_precision_sum = 0.0
        val_recall_sum = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = loss_fn(logits, y)

                n = len(y)
                val_loss_total += float(loss.item()) * n
                val_total += n

                exact, button_acc, precision, recall = batch_metrics(logits, y)
                val_exact_sum += exact * n
                val_button_sum += button_acc * n
                val_precision_sum += precision * n
                val_recall_sum += recall * n

        val_loss = val_loss_total / max(val_total, 1)
        val_exact = val_exact_sum / max(val_total, 1)
        val_button_acc = val_button_sum / max(val_total, 1)
        val_precision = val_precision_sum / max(val_total, 1)
        val_recall = val_recall_sum / max(val_total, 1)

        # Save by a combined score: exact action match + button accuracy.
        score = val_exact + 0.25 * val_button_acc

        print(
            f"[multi_bc] epoch={epoch:02d}/15 "
            f"train_loss={train_loss:.4f} "
            f"train_exact={train_exact:.3f} "
            f"train_button_acc={train_button_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_exact={val_exact:.3f} "
            f"val_button_acc={val_button_acc:.3f} "
            f"val_precision={val_precision:.3f} "
            f"val_recall={val_recall:.3f}"
        )

        if score >= best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "button_names": BUTTON_NAMES,
                "num_buttons": len(BUTTON_NAMES),
                "type": "doomretro_imitation_multibutton_cnn_single_frame",
                "best_score": best_score,
                "data_root": str(DATA_ROOT),
            }, OUT_PATH)
            print(f"[multi_bc] saved {OUT_PATH}")

    print(f"[multi_bc] best_score={best_score:.3f}")


if __name__ == "__main__":
    main()
