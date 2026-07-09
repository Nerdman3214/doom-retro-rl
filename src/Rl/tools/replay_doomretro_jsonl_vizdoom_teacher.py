#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time
import math

import vizdoom as vzd


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


def load_records(path, max_frames=0, skip_leading_noop=True):
    records = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_frames and len(records) >= max_frames:
                break

    if skip_leading_noop:
        first_active = 0
        for i, r in enumerate(records):
            kbd = str(r.get("keyboard_action", ""))
            d = r.get("vizdoom_action_dict") or {}
            if kbd != "no_op" or any(float(v or 0) != 0 for v in d.values()):
                first_active = i
                break
        records = records[first_active:]

    return records


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

    # Strafe, not turning.
    if d.get("MOVE_LEFT", 0) or "strafe_left" in active:
        buttons["MOVE_LEFT"] = 1
    if d.get("MOVE_RIGHT", 0) or "strafe_right" in active:
        buttons["MOVE_RIGHT"] = 1

    if d.get("ATTACK", 0):
        buttons["ATTACK"] = 1
    if d.get("USE", 0) or "use" in active:
        buttons["USE"] = 1

    # Convert unavailable DELTA_VIEW_ANGLE / mouse_dx into TURN_LEFT / TURN_RIGHT.
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

    return [buttons[name] for name in BUTTON_ORDER], buttons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/vizdoom_from_doomretro/doomretro_to_vizdoom_demos.jsonl")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--turn-threshold", type=float, default=0.1)
    ap.add_argument("--invert-turn", action="store_true")
    ap.add_argument("--no-skip-leading-noop", action="store_true")
    ap.add_argument("--reset-on-run-change", action="store_true", default=True)
    args = ap.parse_args()

    records = load_records(
        args.jsonl,
        max_frames=args.max_frames,
        skip_leading_noop=not args.no_skip_leading_noop,
    )

    if not records:
        raise SystemExit("No records loaded.")

    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(int(args.skill))
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(max(2100, len(records) + 300))

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    for var in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH"]:
        if hasattr(vzd.GameVariable, var):
            game.add_available_game_variable(getattr(vzd.GameVariable, var))

    print("[turnfix] IWAD:", args.iwad)
    print("[turnfix] map:", args.map)
    print("[turnfix] skill:", args.skill)
    print("[turnfix] buttons:", BUTTON_ORDER)
    print("[turnfix] records:", len(records))

    game.init()
    game.new_episode()

    frame_dt = 1.0 / max(1.0, args.fps)

    try:
        for i, rec in enumerate(records):
            if game.is_episode_finished():
                print("[turnfix] episode finished")
                break

            current_run = rec.get("run")
            if i > 0:
                previous_run = records[i - 1].get("run")
                if args.reset_on_run_change and current_run != previous_run:
                    print(f"[teacher] run changed {previous_run} -> {current_run}; resetting episode")
                    game.new_episode()

            action, action_dict = rec_to_buttons(
                rec,
                turn_threshold=args.turn_threshold,
                invert_turn=args.invert_turn,
            )

            reward = game.make_action(action, 1)

            if i % 50 == 0:
                state = game.get_state()
                vars_now = list(state.game_variables) if state is not None else []
                active = [BUTTON_ORDER[j] for j, v in enumerate(action) if v]
                print(
                    f"[turnfix] {i:05d} run={rec.get('run')} "
                    f"kbd={rec.get('keyboard_action')} active={active} "
                    f"reward={reward} vars={vars_now}"
                )

            if args.visible:
                time.sleep(frame_dt)

    except KeyboardInterrupt:
        print("[turnfix] stopped by user")

    finally:
        game.close()

    print("[turnfix] done")


if __name__ == "__main__":
    main()
