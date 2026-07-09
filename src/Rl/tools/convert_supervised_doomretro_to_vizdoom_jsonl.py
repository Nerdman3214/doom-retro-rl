#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "supervised_doomretro_v1"
DEFAULT_OUT = ROOT / "data" / "vizdoom_from_doomretro"


ACTION_SCHEMA = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
    "DELTA_VIEW_ANGLE",
    "DELTA_PITCH",
]


def parse_active(label):
    label = (label or "no_op").strip()
    if not label or label == "no_op":
        return set()
    return {x.strip() for x in label.split("+") if x.strip()}


def as_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def find_frame(run_dir, row, idx):
    for key in ["frame_path", "image_path", "screenshot", "frame"]:
        value = row.get(key)
        if value:
            p = Path(value)
            if p.exists():
                return str(p)
            p2 = ROOT / p
            if p2.exists():
                return str(p2)

    frame_num = None
    for key in ["frame_idx", "frame_id", "frame_num", "frame_number", "idx"]:
        if row.get(key) not in [None, ""]:
            try:
                frame_num = int(float(row[key]))
                break
            except Exception:
                pass

    if frame_num is None:
        frame_num = idx

    for ext in ["png", "jpg", "jpeg"]:
        p = run_dir / "frames" / f"frame_{frame_num:06d}.{ext}"
        if p.exists():
            return str(p)

    # Fallback: if rows and files are aligned but numbering differs.
    frames = sorted((run_dir / "frames").glob("frame_*.*"))
    if 0 <= idx < len(frames):
        return str(frames[idx])

    return ""


def row_to_vizdoom_action(row, mouse_scale, max_turn, pitch_scale, max_pitch):
    active = parse_active(row.get("keyboard_action", "no_op"))
    buttons = str(row.get("mouse_buttons", "") or "").lower()

    dx = as_float(row.get("mouse_dx", row.get("dx", row.get("mouse_x", 0.0))))
    dy = as_float(row.get("mouse_dy", row.get("dy", row.get("mouse_y", 0.0))))

    turn = max(-max_turn, min(max_turn, dx * mouse_scale))
    pitch = max(-max_pitch, min(max_pitch, dy * pitch_scale))

    action = {
        "MOVE_FORWARD": 1.0 if "move_forward" in active else 0.0,
        "MOVE_BACKWARD": 1.0 if "move_backward" in active else 0.0,
        "MOVE_LEFT": 1.0 if "strafe_left" in active or "move_left" in active else 0.0,
        "MOVE_RIGHT": 1.0 if "strafe_right" in active or "move_right" in active else 0.0,
        "ATTACK": 1.0 if ("left" in buttons or "attack" in buttons or "shoot" in buttons) else 0.0,
        "USE": 1.0 if "use" in active else 0.0,
        "DELTA_VIEW_ANGLE": turn,
        "DELTA_PITCH": pitch,
    }

    return action, dx, dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mouse-scale", type=float, default=0.10)
    ap.add_argument("--pitch-scale", type=float, default=0.03)
    ap.add_argument("--max-turn", type=float, default=15.0)
    ap.add_argument("--max-pitch", type=float, default=5.0)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    raw_runs = dataset / "raw_runs"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "doomretro_to_vizdoom_demos.jsonl"
    manifest_path = out / "manifest.json"

    if not raw_runs.exists():
        raise SystemExit(f"Missing raw_runs: {raw_runs}")

    total = 0
    runs_written = []

    with jsonl_path.open("w") as fout:
        for run_dir in sorted(raw_runs.iterdir()):
            if not run_dir.is_dir():
                continue

            actions_csv = run_dir / "actions.csv"
            if not actions_csv.exists():
                continue

            rows = list(csv.DictReader(actions_csv.open(newline="")))
            written = 0

            for i, row in enumerate(rows):
                action, raw_dx, raw_dy = row_to_vizdoom_action(
                    row,
                    mouse_scale=args.mouse_scale,
                    max_turn=args.max_turn,
                    pitch_scale=args.pitch_scale,
                    max_pitch=args.max_pitch,
                )

                rec = {
                    "source": "converted_doomretro_supervised",
                    "run": run_dir.name,
                    "t": i,
                    "frame_path": find_frame(run_dir, row, i),
                    "keyboard_action": row.get("keyboard_action", "no_op"),
                    "mouse_dx_raw": raw_dx,
                    "mouse_dy_raw": raw_dy,
                    "mouse_buttons": row.get("mouse_buttons", ""),
                    "vizdoom_schema": ACTION_SCHEMA,
                    "vizdoom_action": [action[k] for k in ACTION_SCHEMA],
                    "vizdoom_action_dict": action,
                }

                fout.write(json.dumps(rec) + "\n")
                total += 1
                written += 1

            if written:
                runs_written.append({"run": run_dir.name, "frames": written})

    manifest = {
        "source_dataset": str(dataset),
        "output_jsonl": str(jsonl_path),
        "schema": ACTION_SCHEMA,
        "runs": runs_written,
        "total_records": total,
        "notes": [
            "This converts Doom Retro teacher clips to a ViZDoom-friendly action dataset.",
            "It does not guarantee identical map physics or exact visual alignment.",
            "Use this for imitation/DAgger bootstrapping in ViZDoom.",
        ],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[convert] wrote: {jsonl_path}")
    print(f"[convert] manifest: {manifest_path}")
    print(f"[convert] runs: {len(runs_written)} records: {total}")


if __name__ == "__main__":
    main()
