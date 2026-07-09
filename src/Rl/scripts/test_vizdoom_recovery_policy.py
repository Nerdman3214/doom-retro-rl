import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import numpy as np

from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button, GameVariable

from navigation.recovery_policy import HardRecoveryPolicy


IWAD_PATH = "/usr/share/games/doom/freedoom1.wad"

ACTIONS = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

BUTTONS = [
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.TURN_LEFT,
    Button.TURN_RIGHT,
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.ATTACK,
    Button.USE,
]

ACTION_TO_BUTTONS = {
    "move_forward": [1, 0, 0, 0, 0, 0, 0, 0],
    "move_backward": [0, 1, 0, 0, 0, 0, 0, 0],
    "turn_left": [0, 0, 1, 0, 0, 0, 0, 0],
    "turn_right": [0, 0, 0, 1, 0, 0, 0, 0],
    "strafe_left": [0, 0, 0, 0, 1, 0, 0, 0],
    "strafe_right": [0, 0, 0, 0, 0, 1, 0, 0],
    "shoot": [0, 0, 0, 0, 0, 0, 1, 0],
    "use": [0, 0, 0, 0, 0, 0, 0, 1],
}


def make_game():
    game = DoomGame()
    game.set_doom_game_path(IWAD_PATH)
    game.set_doom_map("E1M1")

    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.set_screen_format(ScreenFormat.RGB24)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(False)
    game.set_automap_buffer_enabled(False)

    game.set_available_buttons(BUTTONS)

    game.set_available_game_variables([
        GameVariable.POSITION_X,
        GameVariable.POSITION_Y,
        GameVariable.ANGLE,
        GameVariable.HEALTH,
        GameVariable.AMMO2,
    ])

    game.set_episode_timeout(3000)
    game.set_episode_start_time(10)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)

    game.init()
    return game


def get_game_state(game):
    state = game.get_state()

    if state is None:
        return {}

    vars_ = state.game_variables

    return {
        "x": float(vars_[0]),
        "y": float(vars_[1]),
        "angle": float(vars_[2]),
        "health": int(vars_[3]),
        "ammo": int(vars_[4]),
    }


def estimate_obstacles(game):
    """
    ViZDoom depth-buffer obstacle estimate.

    This is intentionally simple:
    - front = close pixels in center
    - left = close pixels on left
    - right = close pixels on right

    If depth is unreliable on your setup, this still falls back to zero pressure.
    """

    state = game.get_state()

    if state is None or state.depth_buffer is None:
        return {"left": 0.0, "front": 0.0, "right": 0.0}

    depth = state.depth_buffer.astype(np.float32)

    h, w = depth.shape

    # In ViZDoom, very low depth values usually mean close geometry.
    # This threshold can be tuned.
    close = depth < 30.0

    top = int(h * 0.25)
    bottom = int(h * 0.80)

    region = close[top:bottom, :]

    left_region = region[:, : int(w * 0.33)]
    front_region = region[:, int(w * 0.33): int(w * 0.66)]
    right_region = region[:, int(w * 0.66):]

    return {
        "left": float(np.mean(left_region)),
        "front": float(np.mean(front_region)),
        "right": float(np.mean(right_region)),
    }


def classify_sensory(game_state, obstacle_info, recovery):
    front = obstacle_info["front"]
    left = obstacle_info["left"]
    right = obstacle_info["right"]

    if recovery.no_movement_steps >= 8:
        return "stuck_or_looping"

    if front >= 0.45:
        return "front_blocked"

    if max(left, right) >= 0.60:
        return "side_rail"

    return "normal_navigation"


def main():
    game = make_game()
    recovery = HardRecoveryPolicy(
        no_movement_threshold_steps=18,
        recovery_steps=28,
        far_wrong_x=700.0,
    )

    target = {
        "name": "level_exit",
        "x": -400.0,
        "y": 1296.0,
    }

    episode = 0

    try:
        while True:
            episode += 1
            game.new_episode()
            recovery.reset()

            print(f"\\n[vizdoom_test] episode={episode}")

            step = 0

            while not game.is_episode_finished():
                step += 1

                game_state = get_game_state(game)
                obstacle_info = estimate_obstacles(game)

                situation = classify_sensory(game_state, obstacle_info, recovery)
                game_state["sensory_situation"] = situation

                # Random policy for testing.
                proposed_action = random.choice([
                    "move_forward",
                    "move_forward",
                    "move_forward",
                    "turn_left",
                    "turn_right",
                    "strafe_left",
                    "strafe_right",
                ])

                action, debug = recovery.action(
                    proposed_action=proposed_action,
                    game_state=game_state,
                    obstacle_info=obstacle_info,
                    target=target,
                )

                game.make_action(ACTION_TO_BUTTONS[action], 4)

                if step % 10 == 0 or debug.get("active"):
                    print(
                        "[viz_recovery] "
                        f"step={step} "
                        f"pos=({game_state.get('x'):.1f},{game_state.get('y'):.1f}) "
                        f"hp={game_state.get('health')} "
                        f"situation={situation} "
                        f"L={obstacle_info['left']:.2f} "
                        f"F={obstacle_info['front']:.2f} "
                        f"R={obstacle_info['right']:.2f} "
                        f"proposed={proposed_action} "
                        f"action={action} "
                        f"active={debug.get('active')} "
                        f"reason={debug.get('reason')} "
                        f"no_move={debug.get('no_movement_steps')}"
                    )

                if game_state.get("health", 100) <= 0:
                    print("[vizdoom_test] died")
                    break

            time.sleep(0.5)

    finally:
        game.close()


if __name__ == "__main__":
    main()
