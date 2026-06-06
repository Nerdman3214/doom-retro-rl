from pathlib import Path
import random
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "vision_dataset" / "raw_human_play" / "vizdoom_wall_recovery_003"
CSV_PATH = DATASET_DIR / "actions.csv"
OUT_PATH = ROOT / "checkpoints" / "recovery_action_prior.pt"

ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def simplify_action(action: str):
    parts = set(str(action).split("+"))

    # For recovery, prioritize geometry escape over shooting/use.
    if "move_backward" in parts:
        return "move_backward"
    if "strafe_left" in parts:
        return "strafe_left"
    if "strafe_right" in parts:
        return "strafe_right"
    if "turn_left" in parts:
        return "turn_left"
    if "turn_right" in parts:
        return "turn_right"
    if "move_forward" in parts:
        return "move_forward"

    # Keep shoot/use rare, but not dominant.
    if "shoot" in parts:
        return "shoot"
    if "use" in parts:
        return "use"

    return "no_op"


class RecoveryDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        rows = []
        for _, row in df.iterrows():
            simple = simplify_action(row["action"])

            # Downsample no_op heavily.
            if simple == "no_op" and random.random() > 0.08:
                continue

            # Downsample shoot because this is recovery, not combat.
            if simple == "shoot" and random.random() > 0.15:
                continue

            # Downsample move_forward a little so turns/strafe matter more.
            if simple == "move_forward" and random.random() > 0.45:
                continue

            if simple not in ACTION_TO_IDX:
                continue

            frame_path = DATASET_DIR / str(row["frame_path"])
            if not frame_path.exists():
                continue

            rows.append({
                "frame_path": row["frame_path"],
                "action": simple,
                "label": ACTION_TO_IDX[simple],
                "trap": row.get("trap", "unknown"),
                "recovery_mode": row.get("recovery_mode", "wall_recovery"),
            })

        self.df = pd.DataFrame(rows)

        self.transform = T.Compose([
            T.Resize((84, 84)),
            T.ToTensor(),
        ])

        print("[recovery_prior] source rows:", len(df))
        print("[recovery_prior] filtered rows:", len(self.df))
        print("\n[recovery_prior] action counts:")
        print(self.df["action"].value_counts())
        print("\n[recovery_prior] recovery mode counts:")
        print(self.df["recovery_mode"].value_counts())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATASET_DIR / row["frame_path"]

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = RecoveryDataset(CSV_PATH)
    if len(dataset) < 100:
        raise SystemExit("Not enough filtered samples. Record more recovery data.")

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(num_actions=len(ACTION_NAMES)).to(device)

    # Class weighting helps because recovery data is imbalanced.
    labels = dataset.df["label"].tolist()
    counts = torch.bincount(torch.tensor(labels), minlength=len(ACTION_NAMES)).float()
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    weights = weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    print("[recovery_prior] device:", device)
    print("[recovery_prior] class counts:", {ACTION_NAMES[i]: int(c) for i, c in enumerate(counts)})

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += len(y)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                pred = logits.argmax(dim=1)

                val_loss_total += float(loss.item()) * len(y)
                val_correct += int((pred == y).sum().item())
                val_total += len(y)

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[recovery_prior] epoch={epoch}/10 "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "action_names": ACTION_NAMES,
                "num_actions": len(ACTION_NAMES),
                "type": "recovery_action_prior",
                "source": str(CSV_PATH),
                "best_val_acc": best_val_acc,
            }, OUT_PATH)
            print(f"[recovery_prior] saved {OUT_PATH}")

    print(f"[recovery_prior] best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
