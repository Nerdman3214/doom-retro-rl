import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}

# Convert compound human labels into one teacher-prior label.
# This order matches "most important action for advice."
COMPOUND_PRIORITY = [
    "shoot",
    "use",
    "move_forward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "move_backward",
]


def find_csvs(raw_root: Path):
    csvs = sorted(raw_root.glob("session_*/actions.csv"))
    csvs += sorted(raw_root.glob("actions.csv"))
    return csvs


def choose_single_action(action_label: str):
    if not action_label or action_label == "no_op":
        return None

    parts = set(action_label.split("+"))

    for name in COMPOUND_PRIORITY:
        if name in parts:
            return name

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default=str(ROOT / "vision_dataset" / "raw_human_play"),
    )
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "vision_dataset_v2" / "teacher_rollouts_balanced"),
    )
    parser.add_argument("--max-no-op", type=int, default=0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    csvs = find_csvs(raw_root)
    if not csvs:
        raise SystemExit(f"No actions.csv files found under {raw_root}")

    if out_root.exists():
        backup = out_root.with_name(out_root.name + "_backup_before_raw_human_convert")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(out_root), str(backup))
        print(f"Backed up old output to {backup}")

    out_root.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    skipped_rows = 0
    action_counts = Counter()

    episode_idx = 1

    for csv_path in csvs:
        session_dir = csv_path.parent
        episode_dir = out_root / f"episode_{episode_idx:06d}"
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = episode_dir / "metadata.jsonl"

        with csv_path.open("r", newline="") as f_in, metadata_path.open("w", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)

            frame_idx = 0

            for row in reader:
                total_rows += 1

                raw_action = row.get("action", "")
                action_name = choose_single_action(raw_action)

                if action_name is None:
                    skipped_rows += 1
                    continue

                frame_rel = row.get("frame_path", "")
                if not frame_rel:
                    skipped_rows += 1
                    continue

                src_frame = session_dir / frame_rel
                if not src_frame.exists():
                    skipped_rows += 1
                    continue

                dst_rel = f"frames/frame_{frame_idx:06d}.png"
                dst_frame = episode_dir / dst_rel
                shutil.copy2(src_frame, dst_frame)

                item = {
                    "frame": dst_rel,
                    "action_index": ACTION_TO_INDEX[action_name],
                    "action_name": action_name,
                    "raw_action": raw_action,
                    "reward": 0.0,
                    "source_csv": str(csv_path),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "angle": row.get("angle"),
                    "health": row.get("health"),
                    "ammo": row.get("ammo"),
                    "kill_count": row.get("kill_count"),
                    "item_count": row.get("item_count"),
                }

                f_out.write(json.dumps(item) + "\n")

                action_counts[action_name] += 1
                kept_rows += 1
                frame_idx += 1

        print(f"Created {episode_dir} with {frame_idx} samples")
        episode_idx += 1

    print("")
    print("Conversion summary")
    print("------------------")
    print(f"raw_root: {raw_root}")
    print(f"out_root: {out_root}")
    print(f"total_rows: {total_rows}")
    print(f"kept_rows: {kept_rows}")
    print(f"skipped_rows: {skipped_rows}")
    print("")
    print("Action counts")
    print("-------------")
    for action, count in action_counts.most_common():
        print(f"{action:15s} {count:8d}")


if __name__ == "__main__":
    main()
