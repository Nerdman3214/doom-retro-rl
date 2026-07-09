#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("vision_dataset")
BACKUPS = sorted(ROOT.glob("noisy_wall_recovery_backup_*"))
if not BACKUPS:
    raise SystemExit("No noisy_wall_recovery_backup_* folder found.")

src_root = BACKUPS[-1]
out_root = Path("vision_dataset/raw_human_play")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

movement_cols = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
]

def truthy(v):
    return str(v).strip() in {"1", "1.0", "true", "True"}

def clean_action(row):
    # Remove shoot/use from wall recovery. We only want escape movement.
    row["shoot"] = "0"
    row["use"] = "0"

    active = [c for c in movement_cols if truthy(row.get(c, "0"))]
    row["action"] = "+".join(active) if active else "no_op"

    row["buttons"] = " ".join(str(int(truthy(row.get(c, "0")))) for c in [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "shoot",
        "use",
    ])
    return row

kept_total = 0

for src in sorted(src_root.glob("session_vizdoom_wall_recovery_*")):
    csv_path = src / "actions.csv"
    frames_src = src / "frames"
    if not csv_path.exists() or not frames_src.exists():
        continue

    out = out_root / f"session_clean_wall_recovery_{stamp}_{src.name}"
    frames_out = out / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        continue

    fieldnames = rows[0].keys()
    out_csv = out / "actions.csv"

    kept = 0
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row in rows:
            row = clean_action(row)

            # Keep only actual movement/turn/strafe/backward recovery labels.
            if row["action"] == "no_op":
                continue

            old_frame = row.get("frame", "")
            candidates = [
                frames_src / f"frame_{int(old_frame):06d}.png" if str(old_frame).isdigit() else None,
                frames_src / f"frame_{kept:06d}.png",
            ]

            src_frame = None
            for c in candidates:
                if c and c.exists():
                    src_frame = c
                    break

            if src_frame is None:
                # Try any frame by original frame number if formatting differs.
                matches = list(frames_src.glob(f"*{old_frame}*"))
                src_frame = matches[0] if matches else None

            if src_frame is None:
                continue

            new_frame = frames_out / f"frame_{kept:06d}.png"
            shutil.copy2(src_frame, new_frame)

            row["frame"] = kept
            w.writerow(row)
            kept += 1

    kept_total += kept
    print(f"[clean] {src.name} -> {out.name} kept={kept}")

print(f"[clean] total kept={kept_total}")
