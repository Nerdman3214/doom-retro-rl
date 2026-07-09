from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "supervised_doomretro_v1"
INDEX_PATH = DATASET / "index" / "supervised_index.csv"
RAW_RUNS = DATASET / "raw_runs"

CACHE_DIR = DATASET / "replay_teacher_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = CACHE_DIR / "teacher_features.npy"
RECORDS_PATH = CACHE_DIR / "teacher_records.json"
META_PATH = CACHE_DIR / "teacher_cache_meta.json"

FEATURE_WIDTH = 32
FEATURE_HEIGHT = 18


KEYBOARD_COLS = ["keyboard_action", "kbd_action", "kbd", "action"]
MOUSE_DX_COLS = ["mouse_dx", "dx", "mouse_x"]
MOUSE_DY_COLS = ["mouse_dy", "dy", "mouse_y"]
BUTTON_COLS = ["mouse_buttons", "buttons", "button"]


def pick_col(row, choices, default=""):
    for col in choices:
        if col in row:
            return row.get(col, default)
    return default


def resolve_frame_path(row):
    path = Path(row.get("frame_path", ""))

    if path.exists():
        return path

    run_name = row.get("run_name", "")
    candidate = RAW_RUNS / run_name / path

    if candidate.exists():
        return candidate

    candidate = RAW_RUNS / run_name / "frames" / path.name

    if candidate.exists():
        return candidate

    return None


def image_feature(path):
    img = Image.open(path).convert("L")
    img = img.resize((FEATURE_WIDTH, FEATURE_HEIGHT), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).reshape(-1)
    return arr


def main():
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index: {INDEX_PATH}")

    records = []
    features = []
    skipped = 0

    with INDEX_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            frame_path = resolve_frame_path(row)

            if frame_path is None:
                skipped += 1
                continue

            try:
                feat = image_feature(frame_path)
            except Exception:
                skipped += 1
                continue

            keyboard_action = pick_col(row, KEYBOARD_COLS, "no_op") or "no_op"
            mouse_dx = pick_col(row, MOUSE_DX_COLS, "0")
            mouse_dy = pick_col(row, MOUSE_DY_COLS, "0")
            mouse_buttons = pick_col(row, BUTTON_COLS, "")

            record = {
                "teacher_index": len(records),
                "run_name": row.get("run_name", ""),
                "frame_index": row.get("frame_index", ""),
                "frame_path": str(frame_path),
                "keyboard_action": keyboard_action,
                "mouse_dx": mouse_dx,
                "mouse_dy": mouse_dy,
                "mouse_buttons": mouse_buttons,
            }

            records.append(record)
            features.append(feat)

            if len(records) % 2500 == 0:
                print(f"[teacher_cache] processed {len(records)} frames...")

    if not features:
        raise SystemExit("No teacher features were built.")

    feature_array = np.stack(features, axis=0)

    np.save(FEATURES_PATH, feature_array)
    RECORDS_PATH.write_text(json.dumps(records, indent=2))

    meta = {
        "index_path": str(INDEX_PATH),
        "features_path": str(FEATURES_PATH),
        "records_path": str(RECORDS_PATH),
        "num_records": len(records),
        "skipped": skipped,
        "feature_width": FEATURE_WIDTH,
        "feature_height": FEATURE_HEIGHT,
        "feature_dim": int(feature_array.shape[1]),
        "dtype": str(feature_array.dtype),
    }

    META_PATH.write_text(json.dumps(meta, indent=2))

    print("=" * 80)
    print("[teacher_cache] DONE")
    print(f"[teacher_cache] records:  {len(records)}")
    print(f"[teacher_cache] skipped:  {skipped}")
    print(f"[teacher_cache] features: {FEATURES_PATH}")
    print(f"[teacher_cache] records:  {RECORDS_PATH}")
    print(f"[teacher_cache] meta:     {META_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()