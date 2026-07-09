import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supervised.train_human_policy import DoomHumanPolicyCNN


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


DEFAULT_THRESHOLDS = {
    "move_forward": 0.60,
    "move_backward": 0.75,
    "turn_left": 0.60,
    "turn_right": 0.65,
    "strafe_left": 0.70,
    "strafe_right": 0.70,
    "shoot": 0.75,
    "use": 0.60,
}


def load_thresholds(path):
    thresholds = dict(DEFAULT_THRESHOLDS)

    p = Path(path)
    if not p.exists():
        return thresholds

    for line in p.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = float(value.strip())
        thresholds[key] = value

    return thresholds


def preprocess_image(path, image_size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def find_csvs(dataset_root):
    dataset_root = Path(dataset_root)
    csvs = sorted(dataset_root.glob("session_*/actions.csv"))
    csvs += sorted(dataset_root.glob("actions.csv"))
    return csvs


def collect_rows(dataset_root, include_no_op=False):
    rows = []

    for csv_path in find_csvs(dataset_root):
        session_dir = csv_path.parent

        with csv_path.open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                action = row.get("action", "")
                frame_rel = row.get("frame_path", "")

                if not frame_rel:
                    continue

                if not include_no_op and action == "no_op":
                    continue

                frame_path = session_dir / frame_rel
                if not frame_path.exists():
                    continue

                row = dict(row)
                row["_frame_path"] = frame_path
                rows.append(row)

    return rows


def clean_impossible_buttons(predicted):
    """
    Remove impossible/conflicting pairs.

    Doom can allow compound actions, but these pairs are usually bad labels:
    - move_forward + move_backward
    - turn_left + turn_right
    - strafe_left + strafe_right
    """

    pred = set(predicted)

    if "move_forward" in pred and "move_backward" in pred:
        pred.remove("move_backward")

    if "turn_left" in pred and "turn_right" in pred:
        pred.remove("turn_right")

    if "strafe_left" in pred and "strafe_right" in pred:
        pred.remove("strafe_right")

    return list(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "supervised_models" / "human_policy_cnn_best.pt"))
    parser.add_argument("--dataset-root", default=str(ROOT / "vision_dataset" / "raw_human_play"))
    parser.add_argument("--thresholds", default=str(ROOT / "supervised_models" / "button_thresholds.txt"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-no-op", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    thresholds = load_thresholds(args.thresholds)

    checkpoint = torch.load(args.model, map_location="cpu")
    image_size = checkpoint.get("image_size", 84)

    model = DoomHumanPolicyCNN(action_dim=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = collect_rows(args.dataset_root, include_no_op=args.include_no_op)
    random.shuffle(rows)
    rows = rows[:args.limit]

    print("[calibrated_test]")
    print(f"thresholds={thresholds}")
    print("")

    with torch.no_grad():
        for row in rows:
            x = preprocess_image(row["_frame_path"], image_size)
            if x is None:
                continue

            logits = model(x)
            probs = torch.sigmoid(logits)[0].numpy()

            predicted = []
            for i, name in enumerate(ACTION_NAMES):
                if probs[i] >= thresholds.get(name, 0.5):
                    predicted.append(name)

            predicted = clean_impossible_buttons(predicted)

            if not predicted:
                predicted = ["no_op"]

            top = sorted(
                [(ACTION_NAMES[i], float(probs[i])) for i in range(len(ACTION_NAMES))],
                key=lambda x: x[1],
                reverse=True,
            )[:4]

            print(f"true={row.get('action'):35s} pred={'+'.join(predicted)}")
            print("  top:", ", ".join(f"{name}={prob:.2f}" for name, prob in top))


if __name__ == "__main__":
    main()
