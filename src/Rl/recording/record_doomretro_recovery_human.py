#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import time
from datetime import datetime

import mss
from PIL import Image

from pynput import keyboard, mouse


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "vision_dataset" / "doomretro_recovery_human"


KEY_TO_ACTION = {
    "w": "move_forward",
    "s": "move_backward",
    "a": "strafe_left",
    "d": "strafe_right",
    "e": "use",
}


class Recorder:
    def __init__(self, args):
        self.args = args
        self.keys_down = set()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.mouse_left = False
        self.running = True

    def on_press(self, key):
        try:
            k = key.char.lower()
        except Exception:
            return
        self.keys_down.add(k)

    def on_release(self, key):
        try:
            k = key.char.lower()
        except Exception:
            return
        self.keys_down.discard(k)

    def on_move(self, x, y):
        # pynput gives absolute position, so we handle movement using previous mouse position.
        if not hasattr(self, "_last_mouse"):
            self._last_mouse = (x, y)
            return

        lx, ly = self._last_mouse
        self.mouse_dx += float(x - lx)
        self.mouse_dy += float(y - ly)
        self._last_mouse = (x, y)

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.mouse_left = bool(pressed)

    def current_keyboard_action(self):
        actions = []
        for key, action in KEY_TO_ACTION.items():
            if key in self.keys_down:
                actions.append(action)
        return "+".join(actions) if actions else "no_op"

    def run_test_capture(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

        with mss.mss() as sct:
            mon = sct.monitors[self.args.monitor]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

        out = OUT_ROOT / "doomretro_recovery_capture_test.jpg"
        img.save(out, quality=90)
        print("[test] saved:", out)
        print("[test] monitor:", mon)

    def run_record(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = OUT_ROOT / f"session_recovery_human_{stamp}"
        frames_dir = session / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        actions_csv = session / "actions.csv"
        meta_txt = session / "meta.txt"

        print("[record] session:", session)
        print("[record] fps:", self.args.fps)
        print("[record] seconds:", self.args.seconds)
        print("[record] monitor:", self.args.monitor)
        print("[record] Focus Doom during countdown.")

        for i in range(5, 0, -1):
            print(f"[record] starting in {i}...")
            time.sleep(1)

        key_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
        )

        key_listener.start()
        mouse_listener.start()

        frame_dt = 1.0 / max(1.0, self.args.fps)
        start = time.time()
        frame_id = 0

        with meta_txt.open("w") as f:
            f.write(f"session={session.name}\n")
            f.write(f"fps={self.args.fps}\n")
            f.write(f"seconds={self.args.seconds}\n")
            f.write(f"monitor={self.args.monitor}\n")
            f.write("type=doomretro_recovery_human\n")

        try:
            with mss.mss() as sct, actions_csv.open("w", newline="") as f:
                mon = sct.monitors[self.args.monitor]

                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_path",
                        "keyboard_action",
                        "mouse_dx",
                        "mouse_dy",
                        "mouse_buttons",
                        "timestamp",
                        "run_name",
                        "source",
                    ],
                )
                writer.writeheader()

                while True:
                    loop_start = time.time()

                    if self.args.seconds > 0 and loop_start - start >= self.args.seconds:
                        break

                    shot = sct.grab(mon)
                    img = Image.frombytes("RGB", shot.size, shot.rgb)

                    frame_name = f"frame_{frame_id:06d}.jpg"
                    frame_path = frames_dir / frame_name
                    img.save(frame_path, quality=self.args.jpeg_quality)

                    keyboard_action = self.current_keyboard_action()
                    mouse_buttons = "left" if self.mouse_left else ""

                    writer.writerow({
                        "frame_path": f"frames/{frame_name}",
                        "keyboard_action": keyboard_action,
                        "mouse_dx": round(self.mouse_dx, 4),
                        "mouse_dy": round(self.mouse_dy, 4),
                        "mouse_buttons": mouse_buttons,
                        "timestamp": round(loop_start - start, 4),
                        "run_name": session.name,
                        "source": "doomretro_recovery_human",
                    })

                    self.mouse_dx = 0.0
                    self.mouse_dy = 0.0

                    frame_id += 1

                    if frame_id % 50 == 0:
                        print(f"[record] saved {frame_id} frames")

                    elapsed = time.time() - loop_start
                    sleep_time = frame_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[record] Ctrl+C received.")
        finally:
            self.running = False
            key_listener.stop()
            mouse_listener.stop()
            print("[record] saved frames:", frame_id)
            print("[record] actions:", actions_csv)
            print("[record] session:", session)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-capture", action="store_true")
    mode.add_argument("--record", action="store_true")

    args = parser.parse_args()

    r = Recorder(args)

    if args.test_capture:
        r.run_test_capture()
    elif args.record:
        r.run_record()


if __name__ == "__main__":
    main()
