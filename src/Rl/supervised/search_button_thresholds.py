import argparse
import csv
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


def f1_score(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-8, precision + recall), precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "supervised_models" / "human_policy_cnn_best.pt"))
    parser.add_argument("--dataset-root", default=str(ROOT / "vision_dataset" / "raw_human_play"))
    parser.add_argument("--out", default=str(ROOT / "supervised_models" / "button_thresholds.txt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    image_size = checkpoint.get("image_size", 84)

    model = DoomHumanPolicyCNN(action_dim=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs_list = []
    labels_list = []

    with torch.no_grad():
        for csv_path in find_csvs(args.dataset_root):
            session_dir = csv_path.parent

            with csv_path.open() as f:
                reader = csv.DictReader(f)

                for row in reader:
                    frame_rel = row.get("frame_path", "")
                    labels = parse_buttons(row.get("buttons", ""))

                    if not frame_rel or labels is None:
                        continue

                    frame_path = session_dir / frame_rel
                    if not frame_path.exists():
                        continue

                    x = preprocess_image(frame_path, image_size)
                    if x is None:
                        continue

                    logits = model(x)
                    probs = torch.sigmoid(logits)[0].numpy()

                    probs_list.append(probs)
                    labels_list.append(labels)

    probs_arr = np.asarray(probs_list, dtype=np.float32)
    labels_arr = np.asarray(labels_list, dtype=np.float32)

    if len(probs_arr) == 0:
        raise SystemExit("No usable samples found.")

    thresholds = []
    lines = []

    print("")
    print("Per-button threshold search")
    print("---------------------------")

    for i, name in enumerate(ACTION_NAMES):
        best = None

        for threshold in np.arange(0.10, 0.96, 0.05):
            pred = (probs_arr[:, i] >= threshold).astype(np.float32)
            y = labels_arr[:, i]

            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())

            f1, precision, recall = f1_score(tp, fp, fn)

            # Slightly prefer precision so the teacher does not press every button.
            score = f1 + 0.05 * precision

            if best is None or score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "score": float(score),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }

        thresholds.append(best["threshold"])
        line = (
            f"{name:14s} threshold={best['threshold']:.2f} "
            f"f1={best['f1']:.3f} precision={best['precision']:.3f} "
            f"recall={best['recall']:.3f} tp={best['tp']} fp={best['fp']} fn={best['fn']}"
        )
        lines.append(line)
        print(line)

    out = Path(args.out)
    out.write_text("\n".join(f"{ACTION_NAMES[i]}={thresholds[i]:.2f}" for i in range(8)) + "\n")

    print("")
    print(f"Saved thresholds to {out}")


if __name__ == "__main__":
    main()
