"""
Doom Retro screen + keyboard + mouse recorder.

Use this for stronger imitation training.

Usage:
1. Start Doom Retro manually.
2. Select easiest difficulty.
3. Load into E1M1.
4. Run this script.
5. Click/focus Doom Retro and play.
6. Press Ctrl+C in terminal to stop/save.

Records:
  - screen frame
  - keyboard action labels
  - mouse_dx / mouse_dy since last frame
  - mouse buttons
  - mouse scroll
"""

import argparse
import csv
import time
from pathlib import Path
from threading import Lock

import cv2
import mss
import numpy as np

try:
    from pynput import keyboard, mouse
except Exception as e:
    raise SystemExit(
        "Missing pynput. Install it with:\n"
        "  pip install pynput\n\n"
        f"Original error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]

pressed_keys = set()
pressed_mouse = set()

mouse_lock = Lock()
last_mouse_x = None
last_mouse_y = None
mouse_dx_accum = 0.0
mouse_dy_accum = 0.0
mouse_scroll_x_accum = 0.0
mouse_scroll_y_accum = 0.0


def key_name(key):
    try:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
    except Exception:
        pass
    return str(key).replace("Key.", "").lower()


def on_key_press(key):
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
        "1": "weapon_1",
        "2": "weapon_2",
        "3": "weapon_3",
        "4": "weapon_4",
        "5": "weapon_5",
        "6": "weapon_6",
        "7": "weapon_7",
    }

    if name in mapping:
        pressed_keys.add(mapping[name])


def on_key_release(key):
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
        "1": "weapon_1",
        "2": "weapon_2",
        "3": "weapon_3",
        "4": "weapon_4",
        "5": "weapon_5",
        "6": "weapon_6",
        "7": "weapon_7",
    }

    if name in mapping:
        pressed_keys.discard(mapping[name])


def on_mouse_move(x, y):
    global last_mouse_x, last_mouse_y, mouse_dx_accum, mouse_dy_accum

    with mouse_lock:
        if last_mouse_x is None or last_mouse_y is None:
            last_mouse_x = x
            last_mouse_y = y
            return

        mouse_dx_accum += float(x - last_mouse_x)
        mouse_dy_accum += float(y - last_mouse_y)

        last_mouse_x = x
        last_mouse_y = y


def on_mouse_click(x, y, button, pressed):
    name = str(button).replace("Button.", "").lower()

    if pressed:
        pressed_mouse.add(name)
    else:
        pressed_mouse.discard(name)


def on_mouse_scroll(x, y, dx, dy):
    global mouse_scroll_x_accum, mouse_scroll_y_accum

    with mouse_lock:
        mouse_scroll_x_accum += float(dx)
        mouse_scroll_y_accum += float(dy)


def pop_mouse_delta():
    global mouse_dx_accum, mouse_dy_accum, mouse_scroll_x_accum, mouse_scroll_y_accum

    with mouse_lock:
        dx = mouse_dx_accum
        dy = mouse_dy_accum
        sx = mouse_scroll_x_accum
        sy = mouse_scroll_y_accum

        mouse_dx_accum = 0.0
        mouse_dy_accum = 0.0
        mouse_scroll_x_accum = 0.0
        mouse_scroll_y_accum = 0.0

    return dx, dy, sx, sy


def current_keyboard_action():
    keys = set(pressed_keys)
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


def current_weapon_action():
    keys = set(pressed_keys)
    weapons = [k for k in sorted(keys) if k.startswith("weapon_")]
    return "+".join(weapons) if weapons else ""


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

    name = args.name or time.strftime("doomretro_mouse_%Y%m%d_%H%M%S")
    out_dir = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation" / name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "actions.csv"
    rows = []

    print("[doomretro_keymouse] Recording Doom Retro screen + keyboard + mouse.")
    print("[doomretro_keymouse] Ctrl+C stops and saves.")
    print(f"[doomretro_keymouse] saving frames to: {frames_dir}")
    print(f"[doomretro_keymouse] csv will save to: {csv_path}")

    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(
        on_move=on_mouse_move,
        on_click=on_mouse_click,
        on_scroll=on_mouse_scroll,
    )

    key_listener.start()
    mouse_listener.start()

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

            print(f"[doomretro_keymouse] capture monitor/crop: {monitor}")
            print("[doomretro_keymouse] recording now... click Doom Retro and play.")

            while True:
                start = time.time()

                img = np.array(sct.grab(monitor))
                frame = img[:, :, :3]

                mouse_dx, mouse_dy, scroll_x, scroll_y = pop_mouse_delta()

                frame_name = f"frame_{frame_idx:06d}.png"
                frame_rel = f"frames/{frame_name}"
                frame_path = frames_dir / frame_name

                cv2.imwrite(str(frame_path), frame)

                keyboard_action = current_keyboard_action()
                weapon_action = current_weapon_action()
                keys = " ".join(sorted(pressed_keys))
                mouse_buttons = " ".join(sorted(pressed_mouse))

                # Mouse left click is shoot for imitation labels.
                shoot_mouse = int("left" in pressed_mouse)
                alt_fire_mouse = int("right" in pressed_mouse)

                rows.append({
                    "frame_path": frame_rel,
                    "keyboard_action": keyboard_action,
                    "weapon_action": weapon_action,
                    "keys": keys,
                    "mouse_buttons": mouse_buttons,
                    "mouse_dx": mouse_dx,
                    "mouse_dy": mouse_dy,
                    "mouse_scroll_x": scroll_x,
                    "mouse_scroll_y": scroll_y,
                    "mouse_left": shoot_mouse,
                    "mouse_right": alt_fire_mouse,
                    "source": "doomretro",
                    "difficulty": "easiest_manual",
                    "timestamp": time.time(),
                })

                if frame_idx % 50 == 0:
                    print(
                        f"[doomretro_keymouse] frame={frame_idx} "
                        f"kbd={keyboard_action} keys={keys} "
                        f"mouse=({mouse_dx:.1f},{mouse_dy:.1f}) "
                        f"buttons={mouse_buttons} saved={frame_path.name}"
                    )

                frame_idx += 1

                elapsed = time.time() - start
                time.sleep(max(0.0, delay - elapsed))

    except KeyboardInterrupt:
        print("\n[doomretro_keymouse] Ctrl+C received. Saving CSV...")

    finally:
        try:
            key_listener.stop()
            mouse_listener.stop()
        except Exception:
            pass

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "keyboard_action",
                    "weapon_action",
                    "keys",
                    "mouse_buttons",
                    "mouse_dx",
                    "mouse_dy",
                    "mouse_scroll_x",
                    "mouse_scroll_y",
                    "mouse_left",
                    "mouse_right",
                    "source",
                    "difficulty",
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"[doomretro_keymouse] saved rows: {len(rows)}")
        print(f"[doomretro_keymouse] csv: {csv_path}")
        print(f"[doomretro_keymouse] frames: {frames_dir}")


if __name__ == "__main__":
    main()
