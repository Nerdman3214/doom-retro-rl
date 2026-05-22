from pathlib import Path
import csv
import shutil


ROOT = Path(__file__).resolve().parents[1]

SOURCE_OBJECT_DIR = ROOT / "vision_dataset_v2" / "objects"

MULTI_DIR = ROOT / "vision_dataset_v2" / "object_multilabel"
IMAGE_DIR = MULTI_DIR / "images"
CSV_PATH = MULTI_DIR / "labels.csv"

LABEL_COLUMNS = [
    "enemy_visible",
    "pickup_health",
    "pickup_ammo",
    "pickup_armor",
    "explosive_barrel",
    "no_important_object",
]

# Only use classes that need boosting.
BOOST_CLASSES = {
    "enemy_visible": 800,
    "explosive_barrel": 100,
}


def read_existing_rows():
    rows = []

    if CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))

    existing_files = {row["filename"] for row in rows}
    return rows, existing_files


def make_empty_label_row(filename):
    row = {"filename": filename}
    for label in LABEL_COLUMNS:
        row[label] = "0"
    return row


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows, existing_files = read_existing_rows()

    added_counts = {label: 0 for label in BOOST_CLASSES}

    for label, max_to_add in BOOST_CLASSES.items():
        source_dir = SOURCE_OBJECT_DIR / label

        if not source_dir.exists():
            print(f"Skipping missing source folder: {source_dir}")
            continue

        image_paths = sorted(
            list(source_dir.glob("*.jpg"))
            + list(source_dir.glob("*.png"))
        )

        for idx, src_path in enumerate(image_paths):
            if added_counts[label] >= max_to_add:
                break

            new_name = f"bootstrap_{label}_{idx:06d}{src_path.suffix.lower()}"

            if new_name in existing_files:
                continue

            dst_path = IMAGE_DIR / new_name
            shutil.copy2(src_path, dst_path)

            row = make_empty_label_row(new_name)
            row[label] = "1"

            # If an important object is visible, no_important_object must be 0.
            row["no_important_object"] = "0"

            rows.append(row)
            existing_files.add(new_name)
            added_counts[label] += 1

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("Added bootstrap rows:")
    for label, count in added_counts.items():
        print(f"{label}: {count}")

    print(f"Updated CSV: {CSV_PATH}")
    print(f"Total rows now: {len(rows)}")


if __name__ == "__main__":
    main()