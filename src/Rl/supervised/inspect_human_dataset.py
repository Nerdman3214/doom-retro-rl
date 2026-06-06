import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_csvs(dataset_root: Path):
    return sorted(dataset_root.glob("session_*/actions.csv")) + sorted(dataset_root.glob("actions.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "vision_dataset" / "raw_human_play"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    csvs = find_csvs(dataset_root)

    if not csvs:
        raise SystemExit(f"No actions.csv files found under {dataset_root}")

    action_counts = Counter()
    total_rows = 0
    missing_frames = 0

    for csv_path in csvs:
        session_dir = csv_path.parent

        with csv_path.open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_rows += 1
                action = row.get("action", "unknown") or "unknown"
                action_counts[action] += 1

                frame_path = row.get("frame_path", "")
                if frame_path:
                    full_frame_path = session_dir / frame_path
                    if not full_frame_path.exists():
                        missing_frames += 1

    print("")
    print("Dataset summary")
    print("---------------")
    print(f"dataset_root: {dataset_root}")
    print(f"sessions/csvs: {len(csvs)}")
    print(f"total_rows: {total_rows}")
    print(f"missing_frames: {missing_frames}")
    print("")
    print("Action counts")
    print("-------------")

    for action, count in action_counts.most_common():
        pct = 100.0 * count / max(1, total_rows)
        print(f"{action:35s} {count:8d}  {pct:6.2f}%")

    print("")
    print("Tip:")
    print("  For a first teacher model, try to avoid too many no_op rows.")
    print("  Good data should include movement, turning, shooting, use, and recovery examples.")


if __name__ == "__main__":
    main()
