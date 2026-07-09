#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import ast
import time

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

ACTION_TO_INDEX = {
    "move_forward": 0,
    "move_backward": 1,
    "turn_left": 2,
    "turn_right": 3,
    "strafe_left": 4,
    "strafe_right": 5,
    "shoot": 6,
    "use": 7,
}


def find_session(name):
    root = Path("vision_dataset/raw_human_play")

    if name == "latest":
        sessions = sorted(root.glob("session_*/actions.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            raise SystemExit("No session_*/actions.csv found.")
        return sessions[0].parent

    if name == "longest":
        best = None
        best_rows = -1
        for p in root.glob("session_*/actions.csv"):
            try:
                rows = sum(1 for _ in p.open()) - 1
            except Exception:
                rows = -1
            if rows > best_rows:
                best_rows = rows
                best = p.parent
        if best is None:
            raise SystemExit("No session_*/actions.csv found.")
        print(f"[replay_session] longest rows={best_rows}")
        return best

    p = Path(name)
    if p.is_dir():
        return p

    p2 = root / name
    if p2.is_dir():
        return p2

    raise SystemExit(f"Could not find session: {name}")


def parse_buttons(row):
    raw = str(row.get("buttons", "")).strip()

    if raw:
        try:
            if raw.startswith("["):
                vals = ast.literal_eval(raw)
                vals = [int(float(x)) for x in vals]
                if len(vals) >= 8:
                    return vals[:8]
        except Exception:
            pass

        # Supports "1 0 0 0 0 0 0 0" or "1,0,0,0,0,0,0,0".
        cleaned = raw.replace(",", " ").replace("[", " ").replace("]", " ")
        parts = [p for p in cleaned.split() if p.strip()]
        if len(parts) >= 8:
            try:
                return [int(float(x)) for x in parts[:8]]
            except Exception:
                pass

    # Fall back to action string.
    buttons = [0] * 8
    action = str(row.get("action", "no_op")).strip()
    if action and action != "no_op":
        for part in action.split("+"):
            part = part.strip()
            if part in ACTION_TO_INDEX:
                buttons[ACTION_TO_INDEX[part]] = 1

    return buttons


def make_game(args, timeout):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    for var in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH"]:
        if hasattr(vzd.GameVariable, var):
            game.add_available_game_variable(getattr(vzd.GameVariable, var))

    return game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="latest", help="latest, longest, or path/name under vision_dataset/raw_human_play")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--skip-leading-noop", action="store_true")
    args = ap.parse_args()

    session = find_session(args.session)
    csv_path = session / "actions.csv"

    rows = list(csv.DictReader(csv_path.open()))
    if args.skip_leading_noop:
        while rows and str(rows[0].get("action", "no_op")) == "no_op":
            rows.pop(0)

    if args.max_frames:
        rows = rows[:args.max_frames]

    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    timeout = max(2100, len(rows) + 300)
    game = make_game(args, timeout=timeout)

    print("[replay_session] session:", session)
    print("[replay_session] rows:", len(rows))
    print("[replay_session] buttons:", BUTTON_ORDER)
    print("[replay_session] iwad:", args.iwad)
    print("[replay_session] map:", args.map)
    print("[replay_session] skill:", args.skill)

    game.init()

    frame_dt = 1.0 / max(1.0, args.fps)

    try:
        for ep in range(args.repeat):
            game.new_episode()
            print(f"[replay_session] episode {ep + 1}/{args.repeat}")

            for i, row in enumerate(rows):
                if game.is_episode_finished():
                    print("[replay_session] episode finished early")
                    break

                buttons = parse_buttons(row)
                reward = game.make_action(buttons, 1)

                if i % 50 == 0:
                    state = game.get_state()
                    vars_now = list(state.game_variables) if state is not None else []
                    active = [BUTTON_ORDER[j] for j, v in enumerate(buttons) if v]
                    print(
                        f"[replay_session] {i:05d} action={row.get('action')} "
                        f"active={active} reward={reward} vars={vars_now}"
                    )

                if args.visible:
                    time.sleep(frame_dt)

    except KeyboardInterrupt:
        print("[replay_session] stopped by user")

    finally:
        game.close()

    print("[replay_session] done")


if __name__ == "__main__":
    main()
