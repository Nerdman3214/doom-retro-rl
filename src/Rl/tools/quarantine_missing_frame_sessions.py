#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path

root = Path("vision_dataset/success_ai_play")
quarantine = Path("vision_dataset/success_ai_play_quarantine_missing_frames")
quarantine.mkdir(parents=True, exist_ok=True)

checked = 0
moved = 0

for d in sorted(root.glob("session_ai_success_*")):
    actions = d / "actions.csv"
    frames = d / "frames"

    if not actions.exists():
        continue

    checked += 1
    missing = []

    try:
        with actions.open(newline="", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        missing.append(f"actions.csv read error: {e}")
        rows = []

    if not frames.exists():
        missing.append("frames folder missing")
    else:
        for row in rows:
            frame_id = row.get("frame", "")
            try:
                frame_id = int(float(frame_id))
            except Exception:
                continue

            for suffix in (f"{frame_id:06d}.png", f"{frame_id}.png"):
                p = frames / f"frame_{suffix}"
                if not p.exists():
                    missing.append(str(p))
                    if len(missing) >= 5:
                        break
            if len(missing) >= 5:
                break

    if missing:
        target = quarantine / d.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(d), str(target))
        moved += 1
        print("[moved]", d.name, "reason:", missing[:3])

print("[done] checked:", checked)
print("[done] moved:", moved)
print("[done] quarantine:", quarantine)
