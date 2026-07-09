"""
Train Doom Retro keyboard+mouse behavior cloning model.

Input:
  vision_dataset_v2/doomretro_keymouse_imitation/doomretro_easiest_keymouse_completion_*/actions.csv

Output:
  checkpoints/doomretro_keymouse_imitation_cnn.pt

Model predicts:
  1. keyboard/mouse button states:
     move_forward, move_backward, strafe_left, strafe_right,
     turn_left, turn_right, use, mouse_left

  2. mouse movement:
     mouse_dx, mouse_dy

This matches FPS play style:
  keyboard movement + mouse aim + mouse shoot
"""

from pathlib import Path
from collections import Counter
import random
import math

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"
OUT_PATH = ROOT / "checkpoints" / "doomretro_keymouse_imitation_cnn.pt"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "use",
    "mouse_left",
]

BUTTON_TO_IDX = {name: i for i, name in enumerate(BUTTON_NAMES)}


def clamp_mouse(v, limit=80.0):
    try:
        v = float(v)
    except Exception:
        return 0.0
    return max(-limit, min(limit, v))


def keyboard_to_buttons(keyboard_action: str, mouse_left: int):
    parts = set(str(keyboard_action).split("+"))
    y = torch.zeros(len(BUTTON_NAMES), dtype=torch.float32)

    for part in parts:
        if part in BUTTON_TO_IDX:
            y[BUTTON_TO_IDX[part]] = 1.0

    if int(mouse_left) > 0:
        y[BUTTON_TO_IDX["mouse_left"]] = 1.0

    return y


class DoomRetroKeyMouseDataset(Dataset):
    def __init__(self):
        self.rows = []

        runs = sorted(DATA_ROOT.glob("doomretro_easiest_keymouse_completion_*"))
        if not runs:
            raise SystemExit(f"No key+mouse runs found in {DATA_ROOT}")

        print(f"[keymouse_bc] found runs: {len(runs)}")

        kept_by_run = Counter()
        action_counts = Counter()
        button_counts = Counter()

        skipped_short = []

        for run in runs:
            csv_path = run / "actions.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            if len(df) < 300:
                skipped_short.append((run.name, len(df)))
                continue

            for _, row in df.iterrows():
                keyboard_action = str(row.get("keyboard_action", "no_op"))
                mouse_left = int(row.get("mouse_left", 0))

                # Downsample fully idle frames.
                mouse_dx = clamp_mouse(row.get("mouse_dx", 0.0))
                mouse_dy = clamp_mouse(row.get("mouse_dy", 0.0))
                is_idle = (
                    keyboard_action == "no_op"
                    and mouse_left == 0
                    and abs(mouse_dx) < 0.5
                    and abs(mouse_dy) < 0.5
                )

                if is_idle and random.random() > 0.12:
                    continue

                frame_path = run / str(row["frame_path"])
                if not frame_path.exists():
                    continue

                buttons = keyboard_to_buttons(keyboard_action, mouse_left)

                # Scale mouse target into roughly [-1, 1].
                mouse_target = torch.tensor(
                    [mouse_dx / 80.0, mouse_dy / 80.0],
                    dtype=torch.float32,
                )

                self.rows.append({
                    "frame_path": frame_path,
                    "buttons": buttons,
                    "mouse": mouse_target,
                    "keyboard_action": keyboard_action,
                    "mouse_left": mouse_left,
                    "run": run.name,
                })

                kept_by_run[run.name] += 1
                action_counts[keyboard_action] += 1

                for name, idx in BUTTON_TO_IDX.items():
                    if buttons[idx].item() > 0.5:
                        button_counts[name] += 1

        if len(self.rows) < 500:
            raise SystemExit("Not enough usable key+mouse rows.")

        print(f"[keymouse_bc] usable rows: {len(self.rows)}")

        if skipped_short:
            print("[keymouse_bc] skipped short runs:")
            for name, n in skipped_short:
                print(f"  {name:<45} rows={n}")

        print("[keymouse_bc] kept rows by run:")
        for run, count in kept_by_run.items():
            print(f"  {run:<45} {count}")

        print("[keymouse_bc] top keyboard actions:")
        for action, count in action_counts.most_common(20):
            print(f"  {action:<35} {count}")

        print("[keymouse_bc] button positives:")
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
        y_buttons = item["buttons"]
        y_mouse = item["mouse"]
        return x, y_buttons, y_mouse


class KeyMouseCNN(nn.Module):
    def __init__(self, num_buttons):
        super().__init__()

        self.backbone = nn.Sequential(
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
        )

        self.button_head = nn.Linear(512, num_buttons)
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.backbone(x)
        button_logits = self.button_head(z)
        mouse_pred = self.mouse_head(z)
        return button_logits, mouse_pred


def button_metrics(logits, y, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    exact = (pred == y).all(dim=1).float().mean().item()
    button_acc = (pred == y).float().mean().item()

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

    dataset = DoomRetroKeyMouseDataset()

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
    model = KeyMouseCNN(num_buttons=len(BUTTON_NAMES)).to(device)

    all_buttons = torch.stack([r["buttons"] for r in dataset.rows])
    positives = all_buttons.sum(dim=0)
    negatives = len(dataset) - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0).to(device)

    print(f"[keymouse_bc] device={device}")
    print("[keymouse_bc] pos_weight:")
    for i, name in enumerate(BUTTON_NAMES):
        print(f"  {name:<15} {float(pos_weight[i].detach().cpu()):.3f}")

    button_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mouse_loss_fn = nn.SmoothL1Loss()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_score = -1.0

    for epoch in range(1, 16):
        model.train()

        total_loss = 0.0
        total_button_loss = 0.0
        total_mouse_loss = 0.0
        total = 0

        train_exact_sum = 0.0
        train_button_acc_sum = 0.0

        for x, y_buttons, y_mouse in train_loader:
            x = x.to(device, non_blocking=True)
            y_buttons = y_buttons.to(device, non_blocking=True)
            y_mouse = y_mouse.to(device, non_blocking=True)

            button_logits, mouse_pred = model(x)

            button_loss = button_loss_fn(button_logits, y_buttons)
            mouse_loss = mouse_loss_fn(mouse_pred, y_mouse)

            # Mouse matters, but button correctness matters more for now.
            loss = button_loss + 0.35 * mouse_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = len(y_buttons)
            total += n
            total_loss += float(loss.item()) * n
            total_button_loss += float(button_loss.item()) * n
            total_mouse_loss += float(mouse_loss.item()) * n

            exact, button_acc, _, _ = button_metrics(button_logits.detach(), y_buttons)
            train_exact_sum += exact * n
            train_button_acc_sum += button_acc * n

        train_loss = total_loss / max(total, 1)
        train_button_loss = total_button_loss / max(total, 1)
        train_mouse_loss = total_mouse_loss / max(total, 1)
        train_exact = train_exact_sum / max(total, 1)
        train_button_acc = train_button_acc_sum / max(total, 1)

        model.eval()

        val_total = 0
        val_loss_total = 0.0
        val_button_loss_total = 0.0
        val_mouse_loss_total = 0.0

        val_exact_sum = 0.0
        val_button_acc_sum = 0.0
        val_precision_sum = 0.0
        val_recall_sum = 0.0
        val_mouse_mae_sum = 0.0

        with torch.no_grad():
            for x, y_buttons, y_mouse in val_loader:
                x = x.to(device, non_blocking=True)
                y_buttons = y_buttons.to(device, non_blocking=True)
                y_mouse = y_mouse.to(device, non_blocking=True)

                button_logits, mouse_pred = model(x)

                button_loss = button_loss_fn(button_logits, y_buttons)
                mouse_loss = mouse_loss_fn(mouse_pred, y_mouse)
                loss = button_loss + 0.35 * mouse_loss

                n = len(y_buttons)
                val_total += n

                val_loss_total += float(loss.item()) * n
                val_button_loss_total += float(button_loss.item()) * n
                val_mouse_loss_total += float(mouse_loss.item()) * n

                exact, button_acc, precision, recall = button_metrics(button_logits, y_buttons)
                val_exact_sum += exact * n
                val_button_acc_sum += button_acc * n
                val_precision_sum += precision * n
                val_recall_sum += recall * n

                # Convert normalized mouse error back into pixel-ish units.
                mouse_mae = (mouse_pred - y_mouse).abs().mean().item() * 80.0
                val_mouse_mae_sum += mouse_mae * n

        val_loss = val_loss_total / max(val_total, 1)
        val_button_loss = val_button_loss_total / max(val_total, 1)
        val_mouse_loss = val_mouse_loss_total / max(val_total, 1)
        val_exact = val_exact_sum / max(val_total, 1)
        val_button_acc = val_button_acc_sum / max(val_total, 1)
        val_precision = val_precision_sum / max(val_total, 1)
        val_recall = val_recall_sum / max(val_total, 1)
        val_mouse_mae = val_mouse_mae_sum / max(val_total, 1)

        # Higher button exact + lower mouse error.
        score = val_exact + 0.25 * val_button_acc - 0.002 * val_mouse_mae

        print(
            f"[keymouse_bc] epoch={epoch:02d}/15 "
            f"train_loss={train_loss:.4f} "
            f"train_btn_loss={train_button_loss:.4f} "
            f"train_mouse_loss={train_mouse_loss:.4f} "
            f"train_exact={train_exact:.3f} "
            f"train_button_acc={train_button_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_btn_loss={val_button_loss:.4f} "
            f"val_mouse_loss={val_mouse_loss:.4f} "
            f"val_exact={val_exact:.3f} "
            f"val_button_acc={val_button_acc:.3f} "
            f"val_precision={val_precision:.3f} "
            f"val_recall={val_recall:.3f} "
            f"val_mouse_mae_px={val_mouse_mae:.2f}"
        )

        if score >= best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "button_names": BUTTON_NAMES,
                "num_buttons": len(BUTTON_NAMES),
                "type": "doomretro_keymouse_imitation_cnn_single_frame",
                "best_score": best_score,
                "data_root": str(DATA_ROOT),
                "mouse_scale": 80.0,
            }, OUT_PATH)
            print(f"[keymouse_bc] saved {OUT_PATH}")

    print(f"[keymouse_bc] best_score={best_score:.3f}")


if __name__ == "__main__":
    main()
