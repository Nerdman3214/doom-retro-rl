from pathlib import Path
import argparse
import csv
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ImageFile.LOAD_TRUNCATED_IMAGES = True


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "supervised_doomretro_v1"
INDEX_DIR = DATASET / "index"
MODEL_DIR = DATASET / "models"
RAW_RUNS = DATASET / "raw_runs"

TRAIN_INDEX = INDEX_DIR / "train_index.csv"
VAL_INDEX = INDEX_DIR / "val_index.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


KEYBOARD_COLS = [
    "keyboard_action",
    "kbd_action",
    "kbd",
    "action",
]

MOUSE_DX_COLS = ["mouse_dx", "dx", "mouse_x"]
MOUSE_DY_COLS = ["mouse_dy", "dy", "mouse_y"]
BUTTON_COLS = ["mouse_buttons", "buttons", "button"]


def pick_col(row, choices, default=""):
    for col in choices:
        if col in row:
            return row.get(col, default)
    return default


def to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def read_csv_rows(path):
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def infer_keyboard_col(fieldnames):
    for col in KEYBOARD_COLS:
        if col in fieldnames:
            return col
    raise RuntimeError(f"Could not find keyboard action column. Columns: {fieldnames}")


def infer_mouse_dx_col(fieldnames):
    for col in MOUSE_DX_COLS:
        if col in fieldnames:
            return col
    return None


def infer_mouse_dy_col(fieldnames):
    for col in MOUSE_DY_COLS:
        if col in fieldnames:
            return col
    return None


def infer_button_col(fieldnames):
    for col in BUTTON_COLS:
        if col in fieldnames:
            return col
    return None


def build_label_map(rows, keyboard_col):
    labels = sorted(set(row.get(keyboard_col, "no_op") or "no_op" for row in rows))
    return {label: i for i, label in enumerate(labels)}


def compute_class_weights(rows, keyboard_col, label_to_id):
    counts = Counter(row.get(keyboard_col, "no_op") or "no_op" for row in rows)
    total = sum(counts.values())
    num_classes = max(1, len(label_to_id))

    weights = torch.ones(num_classes, dtype=torch.float32)

    for label, idx in label_to_id.items():
        count = max(1, counts[label])
        w = total / (num_classes * count)

        # Clamp so rare actions help, but do not explode training.
        w = max(0.35, min(3.0, w))
        weights[idx] = float(w)

    return weights, counts


class DoomSupervisedDataset(Dataset):
    def __init__(
        self,
        rows,
        label_to_id,
        keyboard_col,
        mouse_dx_col,
        mouse_dy_col,
        button_col,
        image_width=160,
        image_height=90,
        mouse_scale=50.0,
        limit=None,
    ):
        if limit is not None:
            rows = rows[:limit]

        self.rows = rows
        self.label_to_id = label_to_id
        self.keyboard_col = keyboard_col
        self.mouse_dx_col = mouse_dx_col
        self.mouse_dy_col = mouse_dy_col
        self.button_col = button_col
        self.image_width = image_width
        self.image_height = image_height
        self.mouse_scale = mouse_scale

    def __len__(self):
        return len(self.rows)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        row = self.rows[idx]

        frame_path = Path(row["frame_path"])

        # Some recorder CSVs store paths like "frames/frame_000123.png".
        # Resolve those relative paths through the clean raw_runs folder.
        if not frame_path.exists():
            run_name = row.get("run_name", "")
            candidate = RAW_RUNS / run_name / frame_path
            if candidate.exists():
                frame_path = candidate

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Missing frame image: original={row.get('frame_path')} "
                f"resolved={frame_path} run={row.get('run_name')}"
            )

        image = self.load_image(frame_path)

        keyboard_label = row.get(self.keyboard_col, "no_op") or "no_op"
        keyboard_id = self.label_to_id.get(keyboard_label, self.label_to_id.get("no_op", 0))

        dx = 0.0
        dy = 0.0

        if self.mouse_dx_col:
            dx = to_float(row.get(self.mouse_dx_col, 0.0))
        if self.mouse_dy_col:
            dy = to_float(row.get(self.mouse_dy_col, 0.0))

        # Clamp extreme flicks so one accidental mouse move does not dominate.
        dx = max(-self.mouse_scale, min(self.mouse_scale, dx))
        dy = max(-self.mouse_scale, min(self.mouse_scale, dy))

        dx_norm = dx / self.mouse_scale
        dy_norm = dy / self.mouse_scale

        buttons = ""
        if self.button_col:
            buttons = row.get(self.button_col, "") or ""

        shoot = 1.0 if "left" in buttons.lower() else 0.0

        try:
            run_weight = float(row.get("run_weight", 1.0))
        except Exception:
            run_weight = 1.0

        sample = {
            "image": image,
            "keyboard_id": torch.tensor(keyboard_id, dtype=torch.long),
            "mouse": torch.tensor([dx_norm, dy_norm], dtype=torch.float32),
            "shoot": torch.tensor([shoot], dtype=torch.float32),
            "weight": torch.tensor(run_weight, dtype=torch.float32),
        }

        return sample


class DoomBCNet(nn.Module):
    def __init__(self, num_keyboard_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        self.keyboard_head = nn.Linear(256, num_keyboard_classes)
        self.mouse_head = nn.Linear(256, 2)
        self.shoot_head = nn.Linear(256, 1)

    def forward(self, x):
        z = self.features(x)
        z = self.shared(z)

        keyboard_logits = self.keyboard_head(z)
        mouse = torch.tanh(self.mouse_head(z))
        shoot_logits = self.shoot_head(z)

        return keyboard_logits, mouse, shoot_logits


def weighted_mean(loss_per_sample, weights):
    while weights.ndim < loss_per_sample.ndim:
        weights = weights.unsqueeze(-1)

    weighted = loss_per_sample * weights
    denom = torch.clamp(weights.sum(), min=1e-6)
    return weighted.sum() / denom


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    class_weights,
    train=True,
    mouse_loss_weight=0.35,
    shoot_loss_weight=0.5,
):
    model.train(train)

    total_loss = 0.0
    total_keyboard_loss = 0.0
    total_mouse_loss = 0.0
    total_shoot_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        keyboard_id = batch["keyboard_id"].to(device, non_blocking=True)
        mouse_target = batch["mouse"].to(device, non_blocking=True)
        shoot_target = batch["shoot"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            keyboard_logits, mouse_pred, shoot_logits = model(images)

            ce_per_sample = F.cross_entropy(
                keyboard_logits,
                keyboard_id,
                weight=class_weights,
                reduction="none",
            )

            mouse_per_sample = F.smooth_l1_loss(
                mouse_pred,
                mouse_target,
                reduction="none",
            ).sum(dim=1)

            shoot_per_sample = F.binary_cross_entropy_with_logits(
                shoot_logits,
                shoot_target,
                reduction="none",
            ).squeeze(1)

            keyboard_loss = weighted_mean(ce_per_sample, weights)
            mouse_loss = weighted_mean(mouse_per_sample, weights)
            shoot_loss = weighted_mean(shoot_per_sample, weights)

            loss = (
                keyboard_loss
                + mouse_loss_weight * mouse_loss
                + shoot_loss_weight * shoot_loss
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        preds = torch.argmax(keyboard_logits, dim=1)
        correct = (preds == keyboard_id).sum().item()
        batch_size = images.size(0)

        total_correct += correct
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_keyboard_loss += keyboard_loss.item() * batch_size
        total_mouse_loss += mouse_loss.item() * batch_size
        total_shoot_loss += shoot_loss.item() * batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_keyboard_loss = total_keyboard_loss / max(1, total_samples)
    avg_mouse_loss = total_mouse_loss / max(1, total_samples)
    avg_shoot_loss = total_shoot_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)

    return {
        "loss": avg_loss,
        "keyboard_loss": avg_keyboard_loss,
        "mouse_loss": avg_mouse_loss,
        "shoot_loss": avg_shoot_loss,
        "keyboard_acc": accuracy,
    }


def save_checkpoint(path, model, optimizer, epoch, label_to_id, config, metrics):
    id_to_label = {v: k for k, v in label_to_id.items()}

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "config": config,
        "metrics": metrics,
    }

    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mouse-scale", type=float, default=50.0)
    parser.add_argument("--mouse-loss-weight", type=float, default=0.35)
    parser.add_argument("--shoot-loss-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not TRAIN_INDEX.exists():
        raise SystemExit(f"Missing train index: {TRAIN_INDEX}")
    if not VAL_INDEX.exists():
        raise SystemExit(f"Missing val index: {VAL_INDEX}")

    train_rows, train_fields = read_csv_rows(TRAIN_INDEX)
    val_rows, val_fields = read_csv_rows(VAL_INDEX)

    if args.limit is not None:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[: max(1, args.limit // 5)]

    keyboard_col = infer_keyboard_col(train_fields)
    mouse_dx_col = infer_mouse_dx_col(train_fields)
    mouse_dy_col = infer_mouse_dy_col(train_fields)
    button_col = infer_button_col(train_fields)

    all_rows = train_rows + val_rows
    label_to_id = build_label_map(all_rows, keyboard_col)

    class_weights, class_counts = compute_class_weights(train_rows, keyboard_col, label_to_id)

    train_ds = DoomSupervisedDataset(
        train_rows,
        label_to_id,
        keyboard_col,
        mouse_dx_col,
        mouse_dy_col,
        button_col,
        image_width=args.image_width,
        image_height=args.image_height,
        mouse_scale=args.mouse_scale,
    )

    val_ds = DoomSupervisedDataset(
        val_rows,
        label_to_id,
        keyboard_col,
        mouse_dx_col,
        mouse_dy_col,
        button_col,
        image_width=args.image_width,
        image_height=args.image_height,
        mouse_scale=args.mouse_scale,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DoomBCNet(num_keyboard_classes=len(label_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = class_weights.to(device)

    config = {
        "train_index": str(TRAIN_INDEX),
        "val_index": str(VAL_INDEX),
        "keyboard_col": keyboard_col,
        "mouse_dx_col": mouse_dx_col,
        "mouse_dy_col": mouse_dy_col,
        "button_col": button_col,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "mouse_scale": args.mouse_scale,
        "mouse_loss_weight": args.mouse_loss_weight,
        "shoot_loss_weight": args.shoot_loss_weight,
        "num_keyboard_classes": len(label_to_id),
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }

    metadata_path = MODEL_DIR / "supervised_bc_metadata.json"
    metadata = {
        "config": config,
        "label_to_id": label_to_id,
        "id_to_label": {v: k for k, v in label_to_id.items()},
        "class_counts": dict(class_counts),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("=" * 80)
    print("[train_bc] Starting supervised behavior cloning")
    print(f"[train_bc] device:       {device}")
    print(f"[train_bc] train rows:   {len(train_ds)}")
    print(f"[train_bc] val rows:     {len(val_ds)}")
    print(f"[train_bc] keyboard col: {keyboard_col}")
    print(f"[train_bc] mouse dx col: {mouse_dx_col}")
    print(f"[train_bc] mouse dy col: {mouse_dy_col}")
    print(f"[train_bc] button col:   {button_col}")
    print(f"[train_bc] classes:      {len(label_to_id)}")
    print("[train_bc] label map:")
    for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
        print(f"  {idx:02d}: {label} count={class_counts.get(label, 0)}")
    print(f"[train_bc] metadata:     {metadata_path}")
    print("=" * 80)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            train=True,
            mouse_loss_weight=args.mouse_loss_weight,
            shoot_loss_weight=args.shoot_loss_weight,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            class_weights,
            train=False,
            mouse_loss_weight=args.mouse_loss_weight,
            shoot_loss_weight=args.shoot_loss_weight,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['keyboard_acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['keyboard_acc']:.3f} "
            f"mouse={val_metrics['mouse_loss']:.4f} "
            f"shoot={val_metrics['shoot_loss']:.4f}"
        )

        last_path = MODEL_DIR / "supervised_bc_last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            label_to_id,
            config,
            {
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = MODEL_DIR / "supervised_bc_best.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                label_to_id,
                config,
                {
                    "train": train_metrics,
                    "val": val_metrics,
                    "best_val_loss": best_val_loss,
                },
            )
            print(f"[train_bc] saved best checkpoint: {best_path}")

    history_path = MODEL_DIR / "supervised_bc_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    print("=" * 80)
    print("[train_bc] DONE")
    print(f"[train_bc] best checkpoint: {MODEL_DIR / 'supervised_bc_best.pt'}")
    print(f"[train_bc] last checkpoint: {MODEL_DIR / 'supervised_bc_last.pt'}")
    print(f"[train_bc] history:         {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()