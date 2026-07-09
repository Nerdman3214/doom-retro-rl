from pathlib import Path
import shutil
import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "vision_dataset" / "raw_frames"
OUT_DIR = ROOT_DIR / "vision_dataset" / "auto_labeled"

CLASSES = [
    "open_path",
    "front_wall",
    "left_wall",
    "right_wall",
    "door_or_button",
    "enemy",
    "pickup",
    "damage_or_death",
    "unclear",
]

for class_name in CLASSES:
    (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def ratio(mask):
    return float(mask.mean()) if mask.size else 0.0


def classify_frame(img):
    """
    Semi-automatic Doom/Freedoom frame labeler.

    This is not meant to be perfect. It creates a first-pass dataset
    so you can review and fix labels much faster.
    """
    if img is None or img.size == 0:
        return "unclear"

    h, w = img.shape[:2]

    # Ignore HUD lower area. Most useful visual info is in top 75%.
    gameplay = img[: int(h * 0.75), :]

    # Regions
    center = gameplay[int(gameplay.shape[0] * 0.25):, int(w * 0.35): int(w * 0.65)]
    left = gameplay[:, : int(w * 0.25)]
    right = gameplay[:, int(w * 0.75):]
    lower_center = gameplay[int(gameplay.shape[0] * 0.55):, int(w * 0.35): int(w * 0.65)]

    # Convert color spaces
    hsv = cv2.cvtColor(gameplay, cv2.COLOR_BGR2HSV)

    # -----------------------------------------------------
    # Damage/death detection
    # -----------------------------------------------------
    # Damage screens usually have strong red tint.
    b, g, r = cv2.split(gameplay)
    red_dominance = ratio((r > 120) & (r > g * 1.35) & (r > b * 1.35))

    # Very dark frames can happen during death/fade.
    gray = cv2.cvtColor(gameplay, cv2.COLOR_BGR2GRAY)
    dark_ratio = ratio(gray < 25)

    if red_dominance > 0.22 or dark_ratio > 0.70:
        return "damage_or_death"

    # -----------------------------------------------------
    # Enemy detection
    # -----------------------------------------------------
    # Freedoom enemies often appear as red/brown/pink/tan clusters.
    # This will catch some walls too, so review this folder manually.
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    red_enemy = (
        ((hue < 12) | (hue > 165))
        & (sat > 70)
        & (val > 50)
    )

    brown_enemy = (
        (hue >= 5)
        & (hue <= 25)
        & (sat > 45)
        & (val > 45)
    )

    enemy_mask = red_enemy | brown_enemy

    # Avoid counting giant wall textures as enemies by requiring clustered blobs.
    enemy_mask_u8 = enemy_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(enemy_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    enemy_blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 40 <= area <= 8000:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(1, bh)
            if 0.25 <= aspect <= 4.0:
                enemy_blobs += 1

    if enemy_blobs >= 1 and ratio(enemy_mask) > 0.005:
        return "enemy"

    # -----------------------------------------------------
    # Pickup detection
    # -----------------------------------------------------
    # Health/ammo/armor/key items are often bright saturated colors.
    bright_green = (
        (hue >= 35)
        & (hue <= 90)
        & (sat > 80)
        & (val > 100)
    )

    bright_blue = (
        (hue >= 90)
        & (hue <= 130)
        & (sat > 80)
        & (val > 100)
    )

    bright_yellow = (
        (hue >= 20)
        & (hue <= 35)
        & (sat > 80)
        & (val > 120)
    )

    pickup_mask = bright_green | bright_blue | bright_yellow

    if ratio(pickup_mask) > 0.01:
        return "pickup"

    # -----------------------------------------------------
    # Wall/open-path detection
    # -----------------------------------------------------
    # Walls tend to have lots of vertical/high-frequency texture.
    # Open floor/path tends to be smoother and lower in the frame.
    center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    lower_gray = cv2.cvtColor(lower_center, cv2.COLOR_BGR2GRAY)

    edges_center = cv2.Canny(center_gray, 60, 140)
    edges_lower = cv2.Canny(lower_gray, 60, 140)

    center_edge_ratio = ratio(edges_center > 0)
    lower_edge_ratio = ratio(edges_lower > 0)

    # Side wall checks
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_edges = cv2.Canny(left_gray, 60, 140)
    right_edges = cv2.Canny(right_gray, 60, 140)

    left_edge_ratio = ratio(left_edges > 0)
    right_edge_ratio = ratio(right_edges > 0)

    if center_edge_ratio > 0.16 and lower_edge_ratio > 0.12:
        return "front_wall"

    if left_edge_ratio > 0.18 and left_edge_ratio > right_edge_ratio + 0.04:
        return "left_wall"

    if right_edge_ratio > 0.18 and right_edge_ratio > left_edge_ratio + 0.04:
        return "right_wall"

    # Door/button rough detection:
    # centered vertical rectangle with strong texture edges.
    mid = gameplay[int(gameplay.shape[0] * 0.25): int(gameplay.shape[0] * 0.80),
                   int(w * 0.40): int(w * 0.60)]
    mid_gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    mid_edges = cv2.Canny(mid_gray, 60, 140)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    vertical_lines = cv2.morphologyEx(mid_edges, cv2.MORPH_OPEN, vertical_kernel)

    if ratio(vertical_lines > 0) > 0.025 and center_edge_ratio > 0.10:
        return "door_or_button"

    # If lower center is relatively smooth, likely walkable space.
    if lower_edge_ratio < 0.10:
        return "open_path"

    return "unclear"


def main():
    images = sorted(
        list(RAW_DIR.glob("*.jpg"))
        + list(RAW_DIR.glob("*.jpeg"))
        + list(RAW_DIR.glob("*.png"))
        + list(RAW_DIR.glob("*.webp"))
    )

    if not images:
        print(f"No images found in {RAW_DIR}")
        return

    counts = {class_name: 0 for class_name in CLASSES}

    for img_path in images:
        img = cv2.imread(str(img_path))
        label = classify_frame(img)

        dst = OUT_DIR / label / img_path.name
        shutil.copy2(img_path, dst)
        counts[label] += 1

    print("\nAuto-label complete.")
    print(f"Input:  {RAW_DIR}")
    print(f"Output: {OUT_DIR}\n")

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")

    print("\nReview these folders manually before training.")
    print("Move wrong labels into the correct folders under vision_dataset/classified.")


if __name__ == "__main__":
    main()