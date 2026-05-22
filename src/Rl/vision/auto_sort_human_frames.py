from pathlib import Path
import shutil
import cv2

from scene_predictor import ScenePredictor


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRS = [
    ROOT / "vision_dataset" / "human_frames" / "freedoom1",
    ROOT / "vision_dataset" / "human_frames" / "freedoom2",
]

CLASSIFIED_DIR = ROOT / "vision_dataset" / "classified"
REVIEW_DIR = ROOT / "vision_dataset" / "needs_review"

# Confidence thresholds.
# These are intentionally stricter for open_path because bad open_path labels
# are one reason the agent gets stuck around spawn/boundary areas.
THRESHOLDS = {
    "enemy": 0.75,
    "front_wall": 0.70,
    "obstacle": 0.65,
    "door_or_button": 0.60,
    "pickup": 0.65,
    "damage_or_death": 0.70,
    "open_path": 0.78,
    "unclear": 0.70,
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{src.parent.name}{src.suffix}"

    shutil.copy2(src, dst)


def main():
    predictor = ScenePredictor()

    all_frames = []
    for input_dir in INPUT_DIRS:
        if input_dir.exists():
            all_frames.extend(
                sorted(
                    p for p in input_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in VALID_EXTS
                )
            )

    if not all_frames:
        print("No frames found.")
        print("Expected frames in:")
        for d in INPUT_DIRS:
            print(f"  - {d}")
        return

    print(f"Found {len(all_frames)} frames.")
    print("Auto-sorting frames...")

    counts = {}

    for i, frame_path in enumerate(all_frames, start=1):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            safe_copy(frame_path, REVIEW_DIR / "bad_read")
            counts["bad_read"] = counts.get("bad_read", 0) + 1
            continue

        result = predictor.predict(frame)
        label = result.get("label", "unclear")
        confidence = float(result.get("confidence", 0.0))

        threshold = THRESHOLDS.get(label, 0.75)

        # Do not auto-accept weak open_path predictions.
        # Bad open_path labels are dangerous because the agent may walk into
        # spawn boundaries, corners, and fake-open areas.
        if confidence >= threshold and label != "unclear":
            target_dir = CLASSIFIED_DIR / label
            bucket = label
        else:
            target_dir = REVIEW_DIR / f"{label}_conf_{confidence:.2f}"
            bucket = "needs_review"

        safe_copy(frame_path, target_dir)
        counts[bucket] = counts.get(bucket, 0) + 1

        if i % 100 == 0:
            print(f"Processed {i}/{len(all_frames)} frames...")

    print("\nDone.")
    print("Counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    print("\nReview these folders manually when your computer can handle it:")
    print(f"  {REVIEW_DIR}")

    print("\nAuto-accepted labels went here:")
    print(f"  {CLASSIFIED_DIR}")


if __name__ == "__main__":
    main()