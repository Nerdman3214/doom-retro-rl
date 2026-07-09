#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from training.train_supervised_bc import DoomBCNet, DoomSupervisedDataset

DATASET = ROOT / "data" / "supervised_doomretro_v1"
TEST_INDEX = DATASET / "index" / "test_index.csv"
MODEL_PATH = DATASET / "models" / "supervised_bc_best.pt"


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_rows(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = safe_torch_load(MODEL_PATH, device)
    label_to_id = ckpt["label_to_id"]
    id_to_label = ckpt["id_to_label"]
    id_to_label = {int(k): v for k, v in id_to_label.items()}
    config = ckpt.get("config", {})

    rows = read_rows(TEST_INDEX)

    ds = DoomSupervisedDataset(
        rows=rows,
        label_to_id=label_to_id,
        keyboard_col="keyboard_action",
        mouse_dx_col="mouse_dx",
        mouse_dy_col="mouse_dy",
        button_col="mouse_buttons",
        image_width=int(config.get("image_width", 160)),
        image_height=int(config.get("image_height", 90)),
        mouse_scale=float(config.get("mouse_scale", 50.0)),
    )

    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = DoomBCNet(num_keyboard_classes=len(label_to_id)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total = 0
    correct = 0
    mouse_abs_sum = 0.0
    mouse_sq_sum = 0.0
    shoot_total = 0
    shoot_correct = 0

    class_total = Counter()
    class_correct = Counter()
    pred_counts = Counter()
    true_counts = Counter()

    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(device, non_blocking=True)
            y = batch["keyboard_id"].to(device, non_blocking=True)
            mouse_true = batch["mouse"].to(device, non_blocking=True)
            shoot_true = batch["shoot"].to(device, non_blocking=True).view(-1)

            keyboard_logits, mouse_pred, shoot_logits = model(images)
            pred = keyboard_logits.argmax(dim=1)

            total += int(y.numel())
            correct += int((pred == y).sum().item())

            diff = mouse_pred - mouse_true
            mouse_abs_sum += float(diff.abs().sum().item())
            mouse_sq_sum += float((diff * diff).sum().item())

            shoot_pred = (torch.sigmoid(shoot_logits).view(-1) >= 0.5).float()
            shoot_total += int(shoot_true.numel())
            shoot_correct += int((shoot_pred == shoot_true).sum().item())

            for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
                true_label = id_to_label[int(t)]
                pred_label = id_to_label[int(p)]

                class_total[true_label] += 1
                true_counts[true_label] += 1
                pred_counts[pred_label] += 1

                if int(t) == int(p):
                    class_correct[true_label] += 1

    print("=" * 80)
    print("[test_eval] model:", MODEL_PATH)
    print("[test_eval] rows:", total)
    print("[test_eval] keyboard_acc:", correct / max(1, total))
    print("[test_eval] mouse_mae_per_value:", mouse_abs_sum / max(1, total * 2))
    print("[test_eval] mouse_mse_per_value:", mouse_sq_sum / max(1, total * 2))
    print("[test_eval] shoot_acc:", shoot_correct / max(1, shoot_total))

    print()
    print("[test_eval] per-class accuracy")
    for label, count in class_total.most_common():
        acc = class_correct[label] / max(1, count)
        print(f"  {label:20s} acc={acc:.3f} count={count}")

    print()
    print("[test_eval] true counts")
    for label, count in true_counts.most_common():
        print(f"  {label:20s} {count}")

    print()
    print("[test_eval] predicted counts")
    for label, count in pred_counts.most_common():
        print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
