#!/usr/bin/env python3
from pathlib import Path
import csv
import random
import argparse
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "supervised_doomretro_v1"
INDEX = DATASET / "index" / "supervised_index.csv"
OUT = ROOT / "debug_teacher_montage.jpg"

def find_frame(row):
    for key in ["frame_path", "image_path", "screenshot", "frame"]:
        p = row.get(key)
        if p:
            path = Path(p)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                return path

    run = row.get("run") or row.get("run_name") or row.get("run_id")
    frame_idx = row.get("frame") or row.get("frame_idx") or row.get("frame_id") or row.get("idx")
    if run and frame_idx:
        try:
            i = int(float(frame_idx))
            candidates = [
                DATASET / "raw_runs" / run / "frames" / f"frame_{i:06d}.jpg",
                DATASET / "raw_runs" / run / "frames" / f"frame_{i:06d}.png",
                DATASET / "raw_runs" / run / f"frame_{i:06d}.jpg",
                DATASET / "raw_runs" / run / f"frame_{i:06d}.png",
            ]
            for c in candidates:
                if c.exists():
                    return c
        except Exception:
            pass

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not INDEX.exists():
        raise SystemExit(f"Missing index: {INDEX}")

    with INDEX.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Index has no rows.")

    picks = random.sample(rows, min(args.samples, len(rows)))

    thumbs = []
    for row in picks:
        frame = find_frame(row)
        if frame is None:
            continue

        img = Image.open(frame).convert("RGB")
        img.thumbnail((320, 180))

        canvas = Image.new("RGB", (320, 230), "black")
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        action = row.get("keyboard_action", "unknown")
        mdx = row.get("mouse_dx", row.get("dx", "0"))
        mdy = row.get("mouse_dy", row.get("dy", "0"))
        buttons = row.get("mouse_buttons", "")
        text = f"{action}\nmouse=({mdx},{mdy}) buttons={buttons}"
        draw.text((6, 184), text, fill="white")

        thumbs.append(canvas)

    if not thumbs:
        raise SystemExit("No frames found. Need to inspect index column names/paths.")

    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    montage = Image.new("RGB", (cols * 320, rows_n * 230), "gray")

    for idx, im in enumerate(thumbs):
        x = (idx % cols) * 320
        y = (idx // cols) * 230
        montage.paste(im, (x, y))

    out = Path(args.out)
    montage.save(out, quality=95)
    print(f"[inspect] wrote {out}")
    print(f"[inspect] rows={len(rows)} samples={len(thumbs)}")

if __name__ == "__main__":
    main()
