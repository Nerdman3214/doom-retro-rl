"""
Simple Doom Retro screen + key recorder.

Usage:
1. Start Doom Retro manually.
2. Select New Game -> easiest difficulty.
3. Load into E1M1.
4. Run this script.
5. Click/focus Doom Retro and play.
6. Press Ctrl+C in the terminal to stop/save.

Saves:
  vision_dataset_v2/doomretro_keyed_imitation/<name>/frames/
  vision_dataset_v2/doomretro_keyed_imitation/<name>/actions.csv
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import mss
import numpy as np

try:
    from pynput import keyboard
except Exception as e:
    raise SystemExit(
        "Missing pynput. Install it with:\n"
        "  pip install pynput\n\n"
        f"Original error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]

pressed = set()


def key_name(key):
    try:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
    except Exception:
        pass

    return str(key).replace("Key.", "").lower()


def on_press(key):
    name = key_name(key)

    mapping = {
        "w": "w",
        "s": "s",
        "a": "a",
        "d": "d",
        "e": "e",
        "space": "space",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "alt_l": "alt",
        "alt_r": "alt",
        "shift_l": "shift",
        "shift_r": "shift",
    }

    if name in mapping:
        pressed.add(mapping[name])


def on_release(key):
    name = key_name(key)

    mapping = {
        "w": "w",
        "s": "s",
        "a": "a",
        "d": "d",
        "e": "e",
        "space": "space",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "alt_l": "alt",
        "alt_r": "alt",
        "shift_l": "shift",
        "shift_r": "shift",
    }

    if name in mapping:
        pressed.discard(mapping[name])


def current_action():
    keys = set(pressed)
    actions = []

    if "w" in keys or "up" in keys:
        actions.append("move_forward")
    if "s" in keys or "down" in keys:
        actions.append("move_backward")

    if "left" in keys:
        actions.append("turn_left")
    if "right" in keys:
        actions.append("turn_right")

    if "a" in keys:
        actions.append("strafe_left")
    if "d" in keys:
        actions.append("strafe_right")

    if "ctrl" in keys:
        actions.append("shoot")
    if "space" in keys or "e" in keys:
        actions.append("use")

    if "alt" in keys and "left" in keys and "strafe_left" not in actions:
        actions.append("strafe_left")
    if "alt" in keys and "right" in keys and "strafe_right" not in actions:
        actions.append("strafe_right")

    return "+".join(actions) if actions else "no_op"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--monitor", type=int, default=1)

    # Optional crop.
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)

    args = parser.parse_args()

    name = args.name or time.strftime("doomretro_keys_%Y%m%d_%H%M%S")
    out_dir = ROOT / "vision_dataset_v2" / "doomretro_keyed_imitation" / name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "actions.csv"
    rows = []

    print("[doomretro_keys] Recording Doom Retro screen + keys.")
    print("[doomretro_keys] Ctrl+C stops and saves.")
    print(f"[doomretro_keys] saving frames to: {frames_dir}")
    print(f"[doomretro_keys] csv will save to: {csv_path}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frame_idx = 0
    delay = 1.0 / max(args.fps, 1)

    try:
        with mss.mss() as sct:
            if args.left or args.top or args.width or args.height:
                monitor = {
                    "left": args.left,
                    "top": args.top,
                    "width": args.width,
                    "height": args.height,
                }
            else:
                monitor = sct.monitors[args.monitor]

            print(f"[doomretro_keys] capture monitor/crop: {monitor}")
            print("[doomretro_keys] recording now... click Doom Retro and play.")

            while True:
                start = time.time()

                img = np.array(sct.grab(monitor))
                frame = img[:, :, :3]  # BGRA/BGR-ish

                frame_name = f"frame_{frame_idx:06d}.png"
                frame_rel = f"frames/{frame_name}"
                frame_path = frames_dir / frame_name

                cv2.imwrite(str(frame_path), frame)

                action = current_action()
                keys = " ".join(sorted(pressed))

                rows.append({
                    "frame_path": frame_rel,
                    "action": action,
                    "keys": keys,
                    "timestamp": time.time(),
                    "source": "doomretro",
                    "difficulty": "easiest_manual",
                })

                if frame_idx % 50 == 0:
                    print(
                        f"[doomretro_keys] frame={frame_idx} "
                        f"action={action} keys={keys} "
                        f"saved={frame_path.name}"
                    )

                frame_idx += 1

                elapsed = time.time() - start
                sleep_for = max(0.0, delay - elapsed)
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[doomretro_keys] Ctrl+C received. Saving CSV...")

    finally:
        try:
            listener.stop()
        except Exception:
            pass

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
                    "keys",
                    "timestamp",
                    "source",
                    "difficulty",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"[doomretro_keys] saved rows: {len(rows)}")
        print(f"[doomretro_keys] csv: {csv_path}")
        print(f"[doomretro_keys] frames: {frames_dir}")


if __name__ == "__main__":
    main()
