import csv
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imitation.action_prior_advisor import ActionPriorAdvisor


ACTION_NAMES = [
    "move_forward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "move_backward",
    "shoot",
    "use",
]


ROLLOUT_ROOT = ROOT / "vision_dataset_v2" / "teacher_rollouts_balanced"


def collect_rows():
    rows = []

    for metadata_path in sorted(ROLLOUT_ROOT.glob("episode_*/metadata.jsonl")):
        episode_dir = metadata_path.parent

        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                frame_path = episode_dir / row["frame"]

                if not frame_path.exists():
                    continue

                row["_frame_path"] = frame_path
                rows.append(row)

    return rows


def main():
    advisor = ActionPriorAdvisor(
        checkpoint_path=ROOT / "checkpoints" / "action_prior_from_teacher_balanced.pt",
        action_names=ACTION_NAMES,
    )

    rows = collect_rows()
    random.seed(42)
    random.shuffle(rows)

    correct = 0
    total = 0

    print("")
    print("[action_prior_prediction_test]")
    print(f"rows={len(rows)}")
    print("")

    for row in rows[:40]:
        frame = cv2.imread(str(row["_frame_path"]), cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = advisor.predict(frame)

        true_action = row["action_name"]
        pred_action = result.get("action_name")
        confidence = result.get("confidence", 0.0)

        if pred_action == true_action:
            correct += 1

        total += 1

        probs = result.get("probs", {})
        top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:4]
        top_text = ", ".join(f"{name}={prob:.2f}" for name, prob in top)

        print(
            f"true={true_action:15s} "
            f"pred={str(pred_action):15s} "
            f"conf={confidence:.2f} "
            f"top={top_text}"
        )

    print("")
    print(f"sample_accuracy={correct}/{total} = {correct / max(1, total):.3f}")


if __name__ == "__main__":
    main()
