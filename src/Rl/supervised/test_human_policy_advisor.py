import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from supervised.human_policy_advisor import HumanPolicyAdvisor


DATASET_ROOT = ROOT / "vision_dataset" / "raw_human_play"


def find_csv():
    csvs = sorted(DATASET_ROOT.glob("session_*/actions.csv"))
    csvs += sorted(DATASET_ROOT.glob("actions.csv"))

    if not csvs:
        raise FileNotFoundError(f"No actions.csv found under {DATASET_ROOT}")

    return csvs[-1]


def main():
    advisor = HumanPolicyAdvisor()

    csv_path = find_csv()
    session_dir = csv_path.parent

    print(f"[advisor_test] csv={csv_path}")
    print("")

    shown = 0

    with csv_path.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("action") == "no_op":
                continue

            frame_path = session_dir / row["frame_path"]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            advice = advisor.advise_from_frame(frame)

            print(
                f"true={row.get('action'):35s} "
                f"teacher={advice['label']:25s} "
                f"confidence={advice['confidence']:.2f}"
            )

            shown += 1
            if shown >= 30:
                break


if __name__ == "__main__":
    main()
