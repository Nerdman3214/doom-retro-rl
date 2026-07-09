#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import shutil
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JSONL = ROOT / "data" / "vizdoom_from_doomretro" / "doomretro_to_vizdoom_demos.jsonl"
DEFAULT_OUT_ROOT = ROOT / "vision_dataset" / "raw_human_play"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

def action_name_from_dict(d):
    active = []
    if d.get("MOVE_FORWARD", 0): active.append("move_forward")
    if d.get("MOVE_BACKWARD", 0): active.append("move_backward")

    # Doom Retro mouse_dx conversion may use DELTA_VIEW_ANGLE.
    turn = float(d.get("DELTA_VIEW_ANGLE", 0) or 0)
    if turn < -0.1: active.append("turn_left")
    if turn > 0.1: active.append("turn_right")

    if d.get("MOVE_LEFT", 0): active.append("strafe_left")
    if d.get("MOVE_RIGHT", 0): active.append("strafe_right")
    if d.get("ATTACK", 0): active.append("shoot")
    if d.get("USE", 0): active.append("use")
    return "+".join(active) if active else "no_op"

def buttons_from_dict(d):
    # Match raw recorder order seen in logs:
    # [move_forward, move_backward, turn_left, turn_right, strafe_left, strafe_right, shoot, use]
    turn = float(d.get("DELTA_VIEW_ANGLE", 0) or 0)
    return [
        int(bool(d.get("MOVE_FORWARD", 0))),
        int(bool(d.get("MOVE_BACKWARD", 0))),
        int(turn < -0.1),
        int(turn > 0.1),
        int(bool(d.get("MOVE_LEFT", 0))),
        int(bool(d.get("MOVE_RIGHT", 0))),
        int(bool(d.get("ATTACK", 0))),
        int(bool(d.get("USE", 0))),
    ]

def safe_copy_or_symlink(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if not src or not Path(src).exists():
        return

    src = Path(src).resolve()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--name-prefix", default="doomretro_converted")
    ap.add_argument("--frame-mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    jsonl = Path(args.jsonl)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not jsonl.exists():
        raise SystemExit(f"Missing JSONL: {jsonl}")

    by_run = defaultdict(list)

    with jsonl.open() as f:
        for n, line in enumerate(f):
            if args.max_records and n >= args.max_records:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            run = rec.get("run") or "unknown_run"
            by_run[run].append(rec)

    print(f"[import] runs found: {len(by_run)}")

    total_rows = 0

    for run, records in sorted(by_run.items()):
        session = out_root / f"{args.name_prefix}_{run}"
        frames_dir = session / "frames"
        session.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        csv_path = session / "actions.csv"

        fields = [
            "frame",
            "action",
            "buttons",
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "shoot",
            "use",
            "x",
            "y",
            "source",
            "source_frame_path",
        ]

        written = 0
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()

            for i, rec in enumerate(records):
                d = rec.get("vizdoom_action_dict") or {}

                # Some converter records may only have the list. Rebuild dict if possible.
                if not d and "vizdoom_schema" in rec and "vizdoom_action" in rec:
                    d = dict(zip(rec["vizdoom_schema"], rec["vizdoom_action"]))

                buttons = buttons_from_dict(d)
                action = action_name_from_dict(d)

                src_frame = rec.get("frame_path") or ""
                ext = ".png"
                if src_frame:
                    suffix = Path(src_frame).suffix.lower()
                    if suffix in [".jpg", ".jpeg", ".png"]:
                        ext = suffix

                dst_frame = frames_dir / f"frame_{i:06d}{ext}"
                safe_copy_or_symlink(src_frame, dst_frame, args.frame_mode)

                row = {
                    "frame": i,
                    "action": action,
                    "buttons": " ".join(str(x) for x in buttons),
                    "move_forward": buttons[0],
                    "move_backward": buttons[1],
                    "turn_left": buttons[2],
                    "turn_right": buttons[3],
                    "strafe_left": buttons[4],
                    "strafe_right": buttons[5],
                    "shoot": buttons[6],
                    "use": buttons[7],
                    "x": rec.get("player_x", ""),
                    "y": rec.get("player_y", ""),
                    "source": "doomretro_converted_jsonl",
                    "source_frame_path": src_frame,
                }
                w.writerow(row)
                written += 1

        total_rows += written
        print(f"[import] wrote {session} rows={written}")

    print(f"[import] total rows={total_rows}")
    print("[import] done")

if __name__ == "__main__":
    main()
