import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from pynput import keyboard
except Exception as e:
    raise SystemExit(
        "pynput is required for keyboard recording. Install it with:\n"
        "  pip install pynput\n"
        f"Original error: {e}"
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


OUTPUT_DIR = ROOT / "vision_dataset" / "human_actions"
FRAME_DIR = OUTPUT_DIR / "frames"
CSV_PATH = OUTPUT_DIR / "actions.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)


pressed_keys = set()


KEY_TO_ACTION = {
    "w": "move_forward",
    "s": "move_backward",
    "a": "strafe_left",
    "d": "strafe_right",
    "left": "turn_left",
    "right": "turn_right",
    "space": "shoot",
    "e": "use",
}


ACTION_PRIORITY = [
    "space",   # shoot
    "e",       # use
    "w",
    "s",
    "a",
    "d",
    "left",
    "right",
]


def key_name(key):
    try:
        return key.char.lower()
    except Exception:
        return str(key).replace("Key.", "").lower()


def on_press(key):
    name = key_name(key)
    pressed_keys.add(name)


def on_release(key):
    name = key_name(key)
    pressed_keys.discard(name)


def current_action_name():
    for key in ACTION_PRIORITY:
        if key in pressed_keys:
            return KEY_TO_ACTION[key]

    # No-op does not exist in your ACTIONS list, so default to turn_left
    # only when no key is pressed. Later we can add a true no_op action.
    return "turn_left"


def extract_screen_frame(env):
    state = env.game.get_state()

    if state is None or state.screen_buffer is None:
        return None

    frame = np.asarray(state.screen_buffer)

    # ViZDoom often returns CHW.
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
        frame = np.transpose(frame[:3], (1, 2, 0))

    if frame.ndim != 3:
        return None

    return frame


def save_frame(frame, frame_id):
    frame_path = FRAME_DIR / f"frame_{frame_id:06d}.png"

    if frame.shape[-1] == 3:
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(frame_path), frame)

    return frame_path


def main():
    print("[record] starting ViZDoom human-action recorder")
    print("[record] controls:")
    print("  W      -> move_forward")
    print("  S      -> move_backward")
    print("  A      -> strafe_left")
    print("  D      -> strafe_right")
    print("  Left   -> turn_left")
    print("  Right  -> turn_right")
    print("  Space  -> shoot")
    print("  E      -> use")
    print("")
    print("[record] IMPORTANT:")
    print("  Keep this terminal focused OR click the ViZDoom window after it opens.")
    print("  Press keys normally. The script sends actions through env.step().")
    print("  Press Ctrl+C in the terminal to stop and save.")
    print("")

    env = VizDoomEnv(visible=True)
    obs, info = env.reset()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    rows = []
    frame_id = 0

    try:
        while True:
            action_name = current_action_name()

            if action_name not in env.ACTIONS:
                action_name = "turn_left"

            action_index = env.ACTIONS.index(action_name)

            obs, reward, done, truncated, info = env.step(action_index)

            frame = extract_screen_frame(env)
            if frame is not None:
                frame_path = save_frame(frame, frame_id)
            else:
                frame_path = None

            game_state = info.get("game_state", {}) or {}
            game_vars = info.get("game_vars", {}) or {}

            rows.append({
                "frame_path": str(frame_path.relative_to(OUTPUT_DIR)) if frame_path else "",
                "action": action_name,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "x": game_state.get("x", game_vars.get("x")),
                "y": game_state.get("y", game_vars.get("y")),
                "angle": game_state.get("angle", game_vars.get("angle")),
                "health": game_state.get("health", game_vars.get("health")),
                "ammo": game_state.get("ammo", game_vars.get("ammo")),
                "kill_count": game_state.get("kill_count", game_vars.get("kill_count")),
                "item_count": game_state.get("item_count", game_vars.get("item_count")),
                "timestamp": time.time(),
            })

            if frame_id % 50 == 0:
                print(
                    f"[record] frame={frame_id} "
                    f"action={action_name} "
                    f"reward={reward:.2f} "
                    f"done={done} truncated={truncated}"
                )

            frame_id += 1

            if done or truncated:
                obs, info = env.reset()

            # Small delay so recording is playable and not too fast.
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("")
        print("[record] stopping and saving dataset...")

    finally:
        try:
            listener.stop()
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
                    "reward",
                    "done",
                    "truncated",
                    "x",
                    "y",
                    "angle",
                    "health",
                    "ammo",
                    "kill_count",
                    "item_count",
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[record] saved rows: {len(rows)}")
        print(f"[record] csv: {CSV_PATH}")
        print(f"[record] frames: {FRAME_DIR}")


if __name__ == "__main__":
    main()
