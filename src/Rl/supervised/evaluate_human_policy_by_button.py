import argparse
import csv
import sys
from collections import defaultdict
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


def parse_buttons(raw):
    parts = str(raw).replace(",", " ").split()
    values = [int(float(x)) for x in parts]
    if len(values) != 8:
        return None
    return np.asarray(values, dtype=np.float32)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "supervised_models" / "human_policy_cnn_best.pt"))
    parser.add_argument("--dataset-root", default=str(ROOT / "vision_dataset" / "raw_human_play"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    image_size = checkpoint.get("image_size", 84)

    model = DoomHumanPolicyCNN(action_dim=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    csvs = find_csvs(args.dataset_root)
    if not csvs:
        raise SystemExit(f"No actions.csv files found under {args.dataset_root}")

    stats = {
        name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for name in ACTION_NAMES
    }

    total = 0
    skipped = 0

    with torch.no_grad():
        for csv_path in csvs:
            session_dir = csv_path.parent

            with csv_path.open() as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if args.limit and total >= args.limit:
                        break

                    frame_rel = row.get("frame_path", "")
                    buttons = parse_buttons(row.get("buttons", ""))

                    if not frame_rel or buttons is None:
                        skipped += 1
                        continue

                    frame_path = session_dir / frame_rel
                    if not frame_path.exists():
                        skipped += 1
                        continue

                    x = preprocess_image(frame_path, image_size)
                    if x is None:
                        skipped += 1
                        continue

                    logits = model(x)
                    probs = torch.sigmoid(logits)[0].numpy()
                    preds = (probs >= args.threshold).astype(np.float32)

                    for i, name in enumerate(ACTION_NAMES):
                        y = int(buttons[i])
                        p = int(preds[i])

                        if y == 1 and p == 1:
                            stats[name]["tp"] += 1
                        elif y == 0 and p == 1:
                            stats[name]["fp"] += 1
                        elif y == 0 and p == 0:
                            stats[name]["tn"] += 1
                        elif y == 1 and p == 0:
                            stats[name]["fn"] += 1

                    total += 1

            if args.limit and total >= args.limit:
                break

    print("")
    print("Button-level evaluation")
    print("-----------------------")
    print(f"model: {args.model}")
    print(f"dataset: {args.dataset_root}")
    print(f"threshold: {args.threshold}")
    print(f"evaluated_rows: {total}")
    print(f"skipped_rows: {skipped}")
    print("")

    for name in ACTION_NAMES:
        tp = stats[name]["tp"]
        fp = stats[name]["fp"]
        tn = stats[name]["tn"]
        fn = stats[name]["fn"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        accuracy = (tp + tn) / max(1, tp + fp + tn + fn)

        print(
            f"{name:14s} "
            f"acc={accuracy:6.3f} "
            f"precision={precision:6.3f} "
            f"recall={recall:6.3f} "
            f"tp={tp:5d} fp={fp:5d} fn={fn:5d}"
        )


if __name__ == "__main__":
    main()
