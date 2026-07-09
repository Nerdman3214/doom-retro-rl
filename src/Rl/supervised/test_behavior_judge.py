import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.behavior_judge_model import BehaviorJudgeCNN


DATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"
CKPT_PATH = ROOT / "checkpoints" / "behavior_judge.pt"


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((84, 84))

    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return torch.tensor(arr, dtype=torch.float32)


def choose_three_frames(frames):
    n = len(frames)

    if n == 1:
        return [frames[0], frames[0], frames[0]]

    early_idx = max(0, int(n * 0.15))
    mid_idx = max(0, int(n * 0.50))
    late_idx = max(0, int(n * 0.85))

    return [frames[early_idx], frames[mid_idx], frames[late_idx]]


def make_clip_tensor(frames):
    selected = choose_three_frames(frames)
    tensors = [preprocess_image(p) for p in selected]
    return torch.cat(tensors, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = BehaviorJudgeCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    label_paths = sorted(DATA_ROOT.glob("clip_*/labels.json"))

    usable = []

    for label_path in label_paths:
        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f)

        if int(label.get("num_frames", 0)) > 300:
            continue

        frames_dir = label_path.parent / "frames"
        frames = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        if not frames:
            continue

        usable.append((label_path, label, frames))

    random.seed(7)
    random.shuffle(usable)

    print(f"[test_behavior_judge_3frame] testing {min(40, len(usable))} clips")

    good_correct = 0
    good_total = 0
    bad_correct = 0
    bad_total = 0

    for label_path, label, frames in usable[:40]:
        x = make_clip_tensor(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        bad_prob = float(probs[0].item())
        good_prob = float(probs[1].item())
        pred = "good" if good_prob >= bad_prob else "bad"

        true_quality = label.get("quality")
        skill = label.get("skill")
        problem = label.get("problem")

        if true_quality == "good":
            good_total += 1
            if pred == "good":
                good_correct += 1
        elif true_quality == "bad":
            bad_total += 1
            if pred == "bad":
                bad_correct += 1

        print(
            f"true={true_quality:4s} pred={pred:4s} "
            f"bad={bad_prob:.3f} good={good_prob:.3f} "
            f"skill={skill:16s} problem={problem} "
            f"clip={label_path.parent.name}"
        )

    print()
    if good_total:
        print(f"[summary] good_acc={good_correct}/{good_total} = {good_correct / good_total:.3f}")
    if bad_total:
        print(f"[summary] bad_acc={bad_correct}/{bad_total} = {bad_correct / bad_total:.3f}")


if __name__ == "__main__":
    main()
