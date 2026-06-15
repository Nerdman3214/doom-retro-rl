from __future__ import annotations

import argparse
import ast
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset" / "raw_human_play"
OUT_PATH = ROOT / "checkpoints" / "vizdoom_raw_behavior_clone.pt"

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


def parse_buttons(row: dict) -> list[float]:
    raw = row.get("buttons", "")

    if raw:
        try:
            vals = ast.literal_eval(raw)
            if isinstance(vals, list) and len(vals) == len(BUTTON_NAMES):
                return [float(x) for x in vals]
        except Exception:
            pass

    action = row.get("action", "") or row.get("keyboard_action", "") or "no_op"
    active = set(action.split("+")) if action and action != "no_op" else set()
    return [1.0 if name in active else 0.0 for name in BUTTON_NAMES]


def find_frame_path(session_dir: Path, row: dict, index: int) -> Path | None:
    for key in ("frame_path", "frame", "image", "image_path"):
        val = row.get(key)
        if val:
            p = Path(val)
            if not p.is_absolute():
                p = session_dir / p
            if p.exists():
                return p

    frames_dir = session_dir / "frames"
    candidates = [
        frames_dir / f"{index:06d}.png",
        frames_dir / f"frame_{index:06d}.png",
        frames_dir / f"{index}.png",
        frames_dir / f"frame_{index}.png",
        frames_dir / f"{index:06d}.jpg",
        frames_dir / f"frame_{index:06d}.jpg",
    ]

    for p in candidates:
        if p.exists():
            return p

    all_frames = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")))
    if index < len(all_frames):
        return all_frames[index]

    return None


class VizDoomRawDataset(Dataset):
    def __init__(self, min_rows: int = 50):
        self.samples: list[tuple[Path, torch.Tensor]] = []

        sessions = []
        for pattern in ("session_*", "scripted_teacher_*", "converted_doomretro_*"):
            sessions.extend(DATA_ROOT.glob(pattern))
        sessions = sorted(set(sessions))

        print(f"[viz_bc] found demo folders: {len(sessions)}")

        skipped = []

        for session in sessions:
            csv_path = session / "actions.csv"
            if not csv_path.exists():
                skipped.append((session.name, "no actions.csv"))
                continue

            with csv_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))

            if len(rows) < min_rows:
                skipped.append((session.name, f"rows={len(rows)}"))
                continue

            kept = 0
            for i, row in enumerate(rows):
                frame_path = find_frame_path(session, row, i)
                if frame_path is None or not frame_path.exists():
                    continue

                y = torch.tensor(parse_buttons(row), dtype=torch.float32)
                self.samples.append((frame_path, y))
                kept += 1

            print(f"[viz_bc] kept {session.name}: rows={len(rows)} samples={kept}")

        if skipped:
            print("[viz_bc] skipped:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")

        if not self.samples:
            raise RuntimeError(f"No samples found in {DATA_ROOT}")

        positives = torch.stack([y for _, y in self.samples]).sum(dim=0)
        print(f"[viz_bc] total samples: {len(self.samples)}")
        print("[viz_bc] button positives:")
        for name, count in zip(BUTTON_NAMES, positives.tolist()):
            print(f"  {name:14s} {int(count)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, y = self.samples[idx]

        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read {frame_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_buttons: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_buttons),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total = 0
    exact = 0
    button_correct = 0
    button_total = 0

    loss_fn = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        pred = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        exact += int((pred == y).all(dim=1).sum().item())
        button_correct += int((pred == y).sum().item())
        button_total += int(y.numel())

    return {
        "loss": total_loss / max(total, 1),
        "exact": exact / max(total, 1),
        "button_acc": button_correct / max(button_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    dataset = VizDoomRawDataset(min_rows=args.min_rows)

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_buttons=len(BUTTON_NAMES)).to(device)

    labels = torch.stack([y for _, y in dataset.samples])
    positives = labels.sum(dim=0)
    negatives = labels.size(0) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=20.0).to(device)

    print("[viz_bc] pos_weight:")
    for name, weight in zip(BUTTON_NAMES, pos_weight.detach().cpu().tolist()):
        print(f"  {name:14s} {weight:.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = -1.0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)

        metrics = evaluate(model, val_loader, device)
        train_loss = total_loss / max(total, 1)

        score = metrics["exact"] + 0.25 * metrics["button_acc"]

        print(
            f"[viz_bc] epoch={epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_exact={metrics['exact']:.3f} "
            f"val_button_acc={metrics['button_acc']:.3f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "button_names": BUTTON_NAMES,
                    "type": "vizdoom_raw_behavior_clone_cnn",
                    "image_size": [84, 84],
                },
                OUT_PATH,
            )
            print(f"[viz_bc] saved {OUT_PATH}")

    print(f"[viz_bc] best_score={best_score:.3f}")


if __name__ == "__main__":
    main()
