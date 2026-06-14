import argparse
import csv
import sys
from pathlib import Path

# Allow running this file directly:
#   python supervised/test_human_policy_predictions.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

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


def preprocess_image(path, image_size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def find_first_csv(dataset_root):
    csvs = sorted(Path(dataset_root).glob("session_*/actions.csv"))
    csvs += sorted(Path(dataset_root).glob("actions.csv"))

    if not csvs:
        raise FileNotFoundError(f"No actions.csv files found under {dataset_root}")

    return csvs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "supervised_models" / "human_policy_cnn_best.pt"))
    parser.add_argument("--dataset-root", default=str(ROOT / "vision_dataset" / "raw_human_play"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    image_size = checkpoint.get("image_size", 84)

    model = DoomHumanPolicyCNN(action_dim=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    csv_path = find_first_csv(args.dataset_root)
    session_dir = csv_path.parent

    print(f"[test_policy] model={args.model}")
    print(f"[test_policy] csv={csv_path}")
    print("")

    shown = 0

    with csv_path.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_rel = row.get("frame_path", "")
            if not frame_rel:
                continue

            frame_path = session_dir / frame_rel
            if not frame_path.exists():
                continue

            x = preprocess_image(frame_path, image_size)

            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].numpy()

            predicted = [
                ACTION_NAMES[i]
                for i, p in enumerate(probs)
                if p >= 0.5
            ]

            if not predicted:
                predicted = ["no_op"]

            prob_text = ", ".join(
                f"{ACTION_NAMES[i]}={probs[i]:.2f}"
                for i in range(len(ACTION_NAMES))
            )

            print(f"true={row.get('action')} | pred={'+'.join(predicted)}")
            print(f"  {prob_text}")

            shown += 1
            if shown >= args.limit:
                break


if __name__ == "__main__":
    main()
