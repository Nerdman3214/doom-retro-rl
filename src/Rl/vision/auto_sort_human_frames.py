from pathlib import Path
import sys
import shutil
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from vision.scene_predictor import ScenePredictor


SOURCE_DIR = ROOT_DIR / "vision_dataset" / "human_frames"
OUT_DIR = ROOT_DIR / "vision_dataset" / "classified"
REVIEW_DIR = ROOT_DIR / "vision_dataset" / "review_needed"

SAMPLE_EVERY = 15

# Do not allow front_wall auto-labeling right now.
# You already have too many wall images.
BLOCKED_LABELS = {"front_wall"}

# Only auto-fill weak classes.
ALLOWED_LABELS = {
    "open_path",
    "obstacle",
    "door_or_button",
    "enemy",
    "unclear",
    "damage_or_death",
}

# Per-class confidence thresholds.
# Lower = more lazy, more noisy.
THRESHOLDS = {
    "open_path": 0.35,
    "obstacle": 0.30,
    "door_or_button": 0.30,
    "enemy": 0.40,
    "unclear": 0.35,
    "damage_or_death": 0.30,
}

# Stop once each class reaches this many images.
TARGET_COUNTS = {
    "open_path": 200,
    "obstacle": 150,
    "door_or_button": 120,
    "enemy": 300,
    "unclear": 150,
    "damage_or_death": 40,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_existing(label):
    class_dir = OUT_DIR / label
    if not class_dir.exists():
        return 0

    return sum(
        1 for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    predictor = ScenePredictor()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    existing_counts = {
        label: count_existing(label)
        for label in TARGET_COUNTS
    }

    print("Starting counts:")
    for label, count in sorted(existing_counts.items()):
        print(f"  {label}: {count}/{TARGET_COUNTS[label]}")

    images = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    )

    if not images:
        print(f"No images found in {SOURCE_DIR}")
        return

    copied = {label: 0 for label in TARGET_COUNTS}
    skipped_front_wall = 0
    skipped_low_conf = 0
    skipped_full = 0

    print(f"\nFound {len(images)} human frames.")
    print(f"Sampling every {SAMPLE_EVERY} frames.")
    print("Blocked labels:", BLOCKED_LABELS)

    for index, img_path in enumerate(images):
        if index % SAMPLE_EVERY != 0:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = predictor.predict(frame)
        label = result["label"]
        confidence = float(result["confidence"])

        if label in BLOCKED_LABELS:
            skipped_front_wall += 1
            continue

        if label not in ALLOWED_LABELS:
            continue

        if existing_counts.get(label, 0) >= TARGET_COUNTS[label]:
            skipped_full += 1
            continue

        threshold = THRESHOLDS.get(label, 0.40)

        if confidence < threshold:
            skipped_low_conf += 1
            continue

        destination_dir = OUT_DIR / label
        destination_dir.mkdir(parents=True, exist_ok=True)

        new_index = existing_counts[label]
        new_name = f"auto_{label}_{new_index:05d}_{img_path.name}"
        destination = destination_dir / new_name

        shutil.copy2(img_path, destination)

        existing_counts[label] += 1
        copied[label] += 1

    print("\nAuto-label complete.")
    print("Copied:")
    for label, count in sorted(copied.items()):
        print(f"  {label}: +{count} now {existing_counts[label]}/{TARGET_COUNTS[label]}")

    print("\nSkipped:")
    print(f"  front_wall blocked: {skipped_front_wall}")
    print(f"  low confidence: {skipped_low_conf}")
    print(f"  class already full: {skipped_full}")

    print("\nNext command:")
    print("python vision/train_scene_classifier.py")


if __name__ == "__main__":
    main()