#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(".").resolve()

EXPORT = ROOT / "supervised_exports" / "e1m1_30of30"
OUT_DIR = ROOT / "data" / "supervised_doomretro_v1" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": EXPORT / "train_sessions.txt",
    "val": EXPORT / "val_sessions.txt",
    "test": EXPORT / "test_sessions.txt",
}

# The normal supervised model has keyboard classes + mouse regression + shoot head.
# So we convert Doom-style turn_left/turn_right into mouse_dx instead of keyboard labels.
TURN_MOUSE = 35.0

KEYBOARD_NAMES = [
    "move_forward",
    "move_backward",
    "strafe_left",
    "strafe_right",
    "use",
]

def action_to_supervised(action_text):
    action_text = str(action_text or "").strip().lower()
    parts = set()

    for part in action_text.replace(",", "+").replace("|", "+").replace(";", "+").split("+"):
        part = part.strip()
        if part and part not in ("no_op", "noop", "none", "0"):
            parts.add(part)

    keyboard = [name for name in KEYBOARD_NAMES if name in parts]
    keyboard_action = "+".join(keyboard) if keyboard else "no_op"

    mouse_dx = 0.0
    if "turn_left" in parts and "turn_right" not in parts:
        mouse_dx = -TURN_MOUSE
    elif "turn_right" in parts and "turn_left" not in parts:
        mouse_dx = TURN_MOUSE

    mouse_dy = 0.0
    mouse_buttons = "left" if "shoot" in parts else ""

    return keyboard_action, mouse_dx, mouse_dy, mouse_buttons

def read_actions_csv(path):
    with path.open("r", newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def build_split(split_name, sessions_file):
    out = OUT_DIR / f"{split_name}_index.csv"
    rows_out = []
    missing_actions = 0
    missing_frames = 0

    sessions = [
        Path(line.strip())
        for line in sessions_file.read_text().splitlines()
        if line.strip()
    ]

    for session_dir in sessions:
        actions_csv = session_dir / "actions.csv"
        if not actions_csv.exists():
            missing_actions += 1
            continue

        for row in read_actions_csv(actions_csv):
            rel_frame = str(row.get("frame_path", "") or "").strip()
            if not rel_frame:
                continue

            frame_path = Path(rel_frame)
            if not frame_path.is_absolute():
                frame_path = session_dir / frame_path

            if not frame_path.exists():
                missing_frames += 1
                continue

            original_action = (
                row.get("action")
                or row.get("buttons")
                or row.get("active")
                or "no_op"
            )

            keyboard_action, mouse_dx, mouse_dy, mouse_buttons = action_to_supervised(original_action)

            rows_out.append({
                "frame_path": str(frame_path),
                "keyboard_action": keyboard_action,
                "mouse_dx": mouse_dx,
                "mouse_dy": mouse_dy,
                "mouse_buttons": mouse_buttons,
                "source_action": original_action,
                "run_name": session_dir.name,
                "split": split_name,
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "z": row.get("z", ""),
                "health": row.get("health", ""),
                "source": row.get("source", ""),
                "wp": row.get("wp", ""),
            })

    fields = [
        "frame_path",
        "keyboard_action",
        "mouse_dx",
        "mouse_dy",
        "mouse_buttons",
        "source_action",
        "run_name",
        "split",
        "x",
        "y",
        "z",
        "health",
        "source",
        "wp",
    ]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"[index] {split_name}: sessions={len(sessions)} rows={len(rows_out)} missing_actions={missing_actions} missing_frames={missing_frames} -> {out}")
    return len(rows_out)

total = 0
for split_name, sessions_file in SPLITS.items():
    if not sessions_file.exists():
        raise SystemExit(f"[stop] missing split file: {sessions_file}")
    total += build_split(split_name, sessions_file)

print("[index] total rows:", total)
print("[index] output dir:", OUT_DIR)
