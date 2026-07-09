#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
from PIL import Image

from vizdoom import DoomGame, Button, GameVariable, ScreenResolution, ScreenFormat


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


def add_var(game, name):
    if hasattr(GameVariable, name):
        game.add_available_game_variable(getattr(GameVariable, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom2.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--visible", action="store_true")
    args = ap.parse_args()

    game = DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_episode_timeout(2100)

    buttons = []
    for name in ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "USE"]:
        add_button(game, buttons, name)

    # Delta buttons are closest to mouse movement. If unavailable, fall back to TURN_LEFT/TURN_RIGHT.
    add_button(game, buttons, "DELTA_VIEW_ANGLE", 20.0)
    add_button(game, buttons, "DELTA_PITCH", 10.0)

    if "DELTA_VIEW_ANGLE" not in buttons:
        add_button(game, buttons, "TURN_LEFT")
        add_button(game, buttons, "TURN_RIGHT")

    for name in ["HEALTH", "AMMO2", "POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "KILLCOUNT", "SECRETCOUNT"]:
        add_var(game, name)

    print("[vizdoom] buttons:", buttons)
    game.init()
    game.new_episode()

    saved = False
    for t in range(120):
        if game.is_episode_finished():
            break

        state = game.get_state()
        if state is not None and not saved:
            img = Image.fromarray(state.screen_buffer)
            out = Path("debug_vizdoom_first_frame.jpg")
            img.save(out)
            print(f"[vizdoom] saved {out.resolve()}")
            saved = True

        action = [0.0] * len(buttons)

        if "MOVE_FORWARD" in buttons:
            action[buttons.index("MOVE_FORWARD")] = 1.0

        if "DELTA_VIEW_ANGLE" in buttons and t % 30 < 12:
            action[buttons.index("DELTA_VIEW_ANGLE")] = 6.0

        reward = game.make_action(action)

        if t % 20 == 0:
            vars_now = game.get_state().game_variables if game.get_state() else []
            print(f"[vizdoom] t={t} reward={reward} vars={vars_now}")

        if args.visible:
            time.sleep(1 / 35)

    game.close()
    print("[vizdoom] OK")


if __name__ == "__main__":
    main()
