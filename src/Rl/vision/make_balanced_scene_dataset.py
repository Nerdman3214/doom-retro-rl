from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "vision_dataset" / "classified"
DST = ROOT / "vision_dataset" / "classified_balanced"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Keep the model from being dominated by enemy/front_wall.
LIMITS = {
    "enemy": 400,
    "front_wall": 400,
    "open_path": 328,
    "unclear": 100,
    "door_or_button": 51,
    "obstacle": 22,
    "damage_or_death": 4,
}

random.seed(42)

if DST.exists():
    shutil.rmtree(DST)

DST.mkdir(parents=True, exist_ok=True)

for class_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    label = class_dir.name

    images = [
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    if not images:
        print(f"Skipping empty class: {label}")
        continue

    limit = LIMITS.get(label, len(images))
    chosen = random.sample(images, min(limit, len(images)))

    out_dir = DST / label
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(chosen):
        out_name = f"{label}_{i:06d}{img.suffix.lower()}"
        shutil.copy2(img, out_dir / out_name)

    print(f"{label}: copied {len(chosen)} / {len(images)}")

print(f"\nBalanced dataset written to: {DST}")