#!/usr/bin/env python3
from pathlib import Path
import csv

ROOT = Path(".").resolve()
RECOVERY_ROOT = ROOT / "vision_dataset" / "doomretro_recovery_human"
OUT = ROOT / "data" / "supervised_doomretro_v1" / "index" / "recovery_human_index.csv"

# Keep real Doom Retro correction signal, but avoid teaching giant flick spikes.
DX_CLAMP = 35.0
DY_CLAMP = 12.0

FIELDS = [
    "frame_path",
    "keyboard_action",
    "mouse_dx",
    "mouse_dy",
    "mouse_buttons",
    "source_action",
    "run_name",
    "split",
    "source",
    "run_weight",
]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

rows_out = []
sessions = sorted(RECOVERY_ROOT.glob("session_recovery_human_*"))

for session in sessions:
    actions = session / "actions.csv"
    frames_dir = session / "frames"

    if not actions.exists() or not frames_dir.exists():
        continue

    with actions.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rel_frame = str(row.get("frame_path", "") or "").strip()
            if not rel_frame:
                continue

            frame_path = session / rel_frame
            if not frame_path.exists():
                continue

            keyboard_action = row.get("keyboard_action", "no_op") or "no_op"
            mouse_buttons = row.get("mouse_buttons", "") or ""

            dx = clamp(to_float(row.get("mouse_dx", 0.0)), -DX_CLAMP, DX_CLAMP)
            dy = clamp(to_float(row.get("mouse_dy", 0.0)), -DY_CLAMP, DY_CLAMP)

            # Project-relative path, so training can open it from the Rl root.
            frame_path_rel = frame_path.relative_to(ROOT)

            rows_out.append({
                "frame_path": str(frame_path_rel),
                "keyboard_action": keyboard_action,
                "mouse_dx": dx,
                "mouse_dy": dy,
                "mouse_buttons": mouse_buttons,
                "source_action": keyboard_action,
                "run_name": session.name,
                "split": "train",
                "source": "doomretro_recovery_human",
                "run_weight": "1.0",
            })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows_out)

print("[recovery_index] sessions:", len(sessions))
print("[recovery_index] rows:", len(rows_out))
print("[recovery_index] wrote:", OUT)
