#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time

from vizdoom import DoomGame, Button, ScreenResolution, ScreenFormat


def add_button(game, buttons, name, max_value=None):
    if not hasattr(Button, name):
        print(f"[skip] Button.{name} not available")
        return
    b = getattr(Button, name)
    if max_value is None:
        game.add_available_button(b)
    else:
        game.add_available_button(b, float(max_value))
    buttons.append(name)


def load_records(path, max_frames=None):
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if max_frames and len(out) >= max_frames:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/vizdoom_from_doomretro/doomretro_to_vizdoom_demos.jsonl")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--fps", type=float, default=35.0)
    ap.add_argument("--max-frames", type=int, default=1000)
    args = ap.parse_args()

    records = load_records(args.jsonl, args.max_frames)
    if not records:
        raise SystemExit("No records loaded.")

    schema = records[0]["vizdoom_schema"]

    game = DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_episode_timeout(max(2100, len(records) + 200))

    buttons = []
    for name in schema:
        if name == "DELTA_VIEW_ANGLE":
            add_button(game, buttons, name, 20.0)
        elif name == "DELTA_PITCH":
            add_button(game, buttons, name, 10.0)
        else:
            add_button(game, buttons, name)

    print("[replay] schema:", schema)
    print("[replay] active ViZDoom buttons:", buttons)

    # If a schema button was skipped by installed ViZDoom, remove that value from actions by name.
    index_map = [schema.index(name) for name in buttons]

    game.init()
    game.new_episode()

    frame_dt = 1.0 / max(1.0, args.fps)

    for i, rec in enumerate(records):
        if game.is_episode_finished():
            print("[replay] episode finished early")
            break

        raw_action = rec["vizdoom_action"]
        action = [raw_action[j] for j in index_map]

        reward = game.make_action(action)

        if i % 50 == 0:
            print(
                f"[replay] {i:05d} run={rec.get('run')} "
                f"kbd={rec.get('keyboard_action')} "
                f"action={action} reward={reward}"
            )

        if args.visible:
            time.sleep(frame_dt)

    game.close()
    print("[replay] done")


if __name__ == "__main__":
    main()
