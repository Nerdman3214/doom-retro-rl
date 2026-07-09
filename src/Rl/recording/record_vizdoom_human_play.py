import csv
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


OUTPUT_DIR = ROOT / "vision_dataset" / "human_play"
FRAME_DIR = OUTPUT_DIR / "frames"
CSV_PATH = OUTPUT_DIR / "actions.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)


def main():
    env = VizDoomEnv(visible=True)

    obs, info = env.reset()

    print("")
    print("Human recording started.")
    print("Play in the ViZDoom window.")
    print("Controls depend on your env action mapping.")
    print("Press Ctrl+C in terminal to stop recording.")
    print("")

    rows = []
    frame_id = 0

    try:
        while True:
            # This script samples the game screen/state while you play manually.
            # It does not control the game.
            state = env.game.get_state()

            if state is None:
                time.sleep(0.05)
                continue

            screen = state.screen_buffer

            if screen is None:
                time.sleep(0.05)
                continue

            # ViZDoom screen_buffer is often CHW.
            if len(screen.shape) == 3 and screen.shape[0] in [1, 3, 4]:
                frame = screen[:3].transpose(1, 2, 0)
            else:
                frame = screen

            frame_path = FRAME_DIR / f"frame_{frame_id:06d}.png"

            # Convert RGB to BGR for cv2.
            if frame.shape[-1] == 3:
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(frame_path), frame)

            game_state = {}

            try:
                game_state = env._get_game_vars(state)
            except Exception:
                game_state = {}

            # For now action is unknown because this passive recorder does not read keys.
            # Later we can add pynput to capture exact keys.
            rows.append({
                "frame_path": str(frame_path.relative_to(OUTPUT_DIR)),
                "action": "unknown",
                "x": game_state.get("x"),
                "y": game_state.get("y"),
                "angle": game_state.get("angle"),
                "health": game_state.get("health"),
                "ammo": game_state.get("ammo"),
                "kill_count": game_state.get("kill_count"),
                "item_count": game_state.get("item_count"),
                "timestamp": time.time(),
            })

            if frame_id % 100 == 0:
                print(f"[record] saved frame={frame_id}")

            frame_id += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("")
        print("Stopping recording...")

    finally:
        env.close()

        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
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

        print(f"Saved {len(rows)} rows to {CSV_PATH}")
        print(f"Frames saved to {FRAME_DIR}")


if __name__ == "__main__":
    main()
