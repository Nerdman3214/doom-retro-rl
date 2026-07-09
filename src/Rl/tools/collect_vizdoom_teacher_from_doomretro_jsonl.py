#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
import vizdoom as vzd


ROOT = Path(__file__).resolve().parents[1]

BUTTON_ORDER = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
]

CSV_BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]


def active_set(kbd):
    kbd = str(kbd or "no_op").strip()
    if not kbd or kbd == "no_op":
        return set()
    return {x.strip() for x in kbd.split("+") if x.strip()}


def get_action_dict(rec):
    d = rec.get("vizdoom_action_dict") or {}
    if d:
        return d

    schema = rec.get("vizdoom_schema") or []
    action = rec.get("vizdoom_action") or []
    return dict(zip(schema, action))


def rec_to_buttons(rec, turn_threshold=0.1, invert_turn=False):
    d = get_action_dict(rec)
    active = active_set(rec.get("keyboard_action"))

    buttons = {name: 0 for name in BUTTON_ORDER}

    if d.get("MOVE_FORWARD", 0) or "move_forward" in active:
        buttons["MOVE_FORWARD"] = 1
    if d.get("MOVE_BACKWARD", 0) or "move_backward" in active:
        buttons["MOVE_BACKWARD"] = 1

    if d.get("MOVE_LEFT", 0) or "strafe_left" in active:
        buttons["MOVE_LEFT"] = 1
    if d.get("MOVE_RIGHT", 0) or "strafe_right" in active:
        buttons["MOVE_RIGHT"] = 1

    if d.get("ATTACK", 0):
        buttons["ATTACK"] = 1
    if d.get("USE", 0) or "use" in active:
        buttons["USE"] = 1

    # Convert Doom Retro mouse/DELTA turning into ViZDoom turn buttons.
    turn = 0.0
    for key in ["DELTA_VIEW_ANGLE", "mouse_dx_raw", "mouse_dx"]:
        try:
            if key in d:
                turn = float(d.get(key) or 0)
                break
            if key in rec:
                turn = float(rec.get(key) or 0)
                break
        except Exception:
            pass

    if invert_turn:
        turn *= -1.0

    if turn < -turn_threshold or "turn_left" in active:
        buttons["TURN_LEFT"] = 1
    elif turn > turn_threshold or "turn_right" in active:
        buttons["TURN_RIGHT"] = 1

    return [buttons[name] for name in BUTTON_ORDER]


def action_name(buttons):
    active = [CSV_BUTTON_NAMES[i] for i, v in enumerate(buttons) if int(v)]
    return "+".join(active) if active else "no_op"


def save_frame(screen_buffer, path):
    arr = np.asarray(screen_buffer)

    # ViZDoom RGB24 is usually HWC, but handle CHW just in case.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def load_by_run(jsonl, max_records=0):
    by_run = defaultdict(list)
    total = 0

    with Path(jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            run = rec.get("run") or "unknown_run"
            by_run[run].append(rec)
            total += 1
            if max_records and total >= max_records:
                break

    return by_run


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(int(args.skill))
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    for var in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH"]:
        if hasattr(vzd.GameVariable, var):
            game.add_available_game_variable(getattr(vzd.GameVariable, var))

    return game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/vizdoom_from_doomretro/doomretro_to_vizdoom_demos.jsonl")
    ap.add_argument("--out-root", default="vision_dataset/raw_human_play")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--fps", type=float, default=35.0)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--episode-timeout", type=int, default=2100)
    ap.add_argument("--turn-threshold", type=float, default=0.1)
    ap.add_argument("--invert-turn", action="store_true")
    ap.add_argument("--max-consecutive-noop", type=int, default=25)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    by_run = load_by_run(args.jsonl, args.max_records)
    run_items = sorted(by_run.items())

    if args.max_runs:
        run_items = run_items[:args.max_runs]

    print(f"[collect_teacher] runs={len(run_items)}")
    print(f"[collect_teacher] iwad={args.iwad} map={args.map} skill={args.skill}")

    game = make_game(args)
    game.init()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_saved = 0

    try:
        for run_name, records in run_items:
            session = out_root / f"teacher_doomretro_{stamp}_{run_name}"
            frames_dir = session / "frames"
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
                "angle",
                "health",
            ]

            print(f"[collect_teacher] run={run_name} records={len(records)} -> {session}")
            game.new_episode()

            saved = 0
            consecutive_noop = 0

            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                for rec in records:
                    if game.is_episode_finished():
                        print(f"[collect_teacher] episode finished early for {run_name}")
                        break

                    state = game.get_state()
                    if state is None:
                        break

                    buttons = rec_to_buttons(
                        rec,
                        turn_threshold=args.turn_threshold,
                        invert_turn=args.invert_turn,
                    )
                    name = action_name(buttons)

                    if name == "no_op":
                        consecutive_noop += 1
                    else:
                        consecutive_noop = 0

                    # Skip very long idle tails so the model does not learn to stand still.
                    if consecutive_noop > args.max_consecutive_noop:
                        game.make_action(buttons, 1)
                        continue

                    frame_path = frames_dir / f"frame_{saved:06d}.png"
                    save_frame(state.screen_buffer, frame_path)

                    vars_now = list(state.game_variables) if state.game_variables is not None else []
                    x = vars_now[0] if len(vars_now) > 0 else ""
                    y = vars_now[1] if len(vars_now) > 1 else ""
                    angle = vars_now[2] if len(vars_now) > 2 else ""
                    health = vars_now[3] if len(vars_now) > 3 else ""

                    row = {
                        "frame": saved,
                        "action": name,
                        "buttons": " ".join(str(int(v)) for v in buttons),
                        "move_forward": buttons[0],
                        "move_backward": buttons[1],
                        "turn_left": buttons[2],
                        "turn_right": buttons[3],
                        "strafe_left": buttons[4],
                        "strafe_right": buttons[5],
                        "shoot": buttons[6],
                        "use": buttons[7],
                        "x": x,
                        "y": y,
                        "angle": angle,
                        "health": health,
                    }
                    writer.writerow(row)
                    saved += 1
                    total_saved += 1

                    game.make_action(buttons, 1)

                    if args.visible:
                        time.sleep(1.0 / max(1.0, args.fps))

            print(f"[collect_teacher] saved rows={saved} csv={csv_path}")

    except KeyboardInterrupt:
        print("[collect_teacher] stopped by user")

    finally:
        game.close()

    print(f"[collect_teacher] total_saved={total_saved}")


if __name__ == "__main__":
    main()
