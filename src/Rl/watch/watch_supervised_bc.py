from pathlib import Path
import argparse
import json
import time
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

import torch

import mss
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button

from training.train_supervised_bc import DoomBCNet


DATASET = ROOT / "data" / "supervised_doomretro_v1"
DEFAULT_MODEL = DATASET / "models" / "supervised_bc_best.pt"


KEY_MAP = {
    "move_forward": "w",
    "move_backward": "s",
    "strafe_left": "a",
    "strafe_right": "d",
    "use": "e",
}


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path, device):
    ckpt = safe_torch_load(model_path, device)

    label_to_id = ckpt["label_to_id"]
    id_to_label = ckpt["id_to_label"]
    config = ckpt["config"]

    # JSON sometimes converts integer keys to strings.
    id_to_label = {int(k): v for k, v in id_to_label.items()}

    model = DoomBCNet(num_keyboard_classes=len(label_to_id)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, label_to_id, id_to_label, config


def preprocess_image(img, width, height):
    img = img.convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def parse_keyboard_action(label):
    if not label or label == "no_op":
        return set()

    keys = set()

    parts = label.split("+")
    for part in parts:
        part = part.strip()
        key = KEY_MAP.get(part)
        if key:
            keys.add(key)

    return keys


def release_all_keys(keyboard, pressed_keys):
    for key in list(pressed_keys):
        try:
            keyboard.release(key)
        except Exception:
            pass
    pressed_keys.clear()


def update_keys(keyboard, pressed_keys, wanted_keys):
    for key in list(pressed_keys):
        if key not in wanted_keys:
            try:
                keyboard.release(key)
            except Exception:
                pass
            pressed_keys.remove(key)

    for key in wanted_keys:
        if key not in pressed_keys:
            try:
                keyboard.press(key)
                pressed_keys.add(key)
            except Exception:
                pass


def read_player_position(state_json):
    if not state_json:
        return None

    path = Path(state_json)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    # Supported formats:
    # {"player_x": 123, "player_y": 456}
    # {"x": 123, "y": 456}
    # {"player": {"x": 123, "y": 456}}
    # {"position": {"x": 123, "y": 456}}

    if "player_x" in data and "player_y" in data:
        return float(data["player_x"]), float(data["player_y"])

    if "x" in data and "y" in data:
        return float(data["x"]), float(data["y"])

    if isinstance(data.get("player"), dict):
        p = data["player"]
        if "x" in p and "y" in p:
            return float(p["x"]), float(p["y"])

    if isinstance(data.get("position"), dict):
        p = data["position"]
        if "x" in p and "y" in p:
            return float(p["x"]), float(p["y"])

    return None


def maybe_press_reset(keyboard, reset_key):
    if not reset_key:
        return

    key_name = reset_key.lower().strip()

    special = {
        "f5": Key.f5,
        "f6": Key.f6,
        "f7": Key.f7,
        "f8": Key.f8,
        "f9": Key.f9,
        "f10": Key.f10,
        "f11": Key.f11,
        "f12": Key.f12,
        "esc": Key.esc,
        "escape": Key.esc,
        "enter": Key.enter,
        "space": Key.space,
    }

    key = special.get(key_name, key_name)

    keyboard.press(key)
    time.sleep(0.08)
    keyboard.release(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)

    parser.add_argument("--mouse-gain", type=float, default=0.75)
    parser.add_argument("--shoot-threshold", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)

    parser.add_argument("--state-json", type=str, default="")
    parser.add_argument("--stuck-seconds", type=float, default=3.0)
    parser.add_argument("--stuck-min-delta", type=float, default=1.0)
    parser.add_argument("--reset-key", type=str, default="f9")
    parser.add_argument("--reset-wait", type=float, default=1.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_to_id, id_to_label, config = load_model(Path(args.model), device)

    image_width = int(config.get("image_width", 160))
    image_height = int(config.get("image_height", 90))
    mouse_scale = float(config.get("mouse_scale", 50.0))

    keyboard = KeyboardController()
    mouse = MouseController()
    pressed_keys = set()

    frame_dt = 1.0 / max(1.0, args.fps)

    print("=" * 80)
    print("[watch_bc] Supervised Doom Retro watcher")
    print(f"[watch_bc] model:          {args.model}")
    print(f"[watch_bc] device:         {device}")
    print(f"[watch_bc] fps:            {args.fps}")
    print(f"[watch_bc] image size:     {image_width}x{image_height}")
    print(f"[watch_bc] mouse_scale:    {mouse_scale}")
    print(f"[watch_bc] mouse_gain:     {args.mouse_gain}")
    print(f"[watch_bc] dry_run:        {args.dry_run}")
    print(f"[watch_bc] state_json:     {args.state_json or 'none'}")
    print(f"[watch_bc] stuck_seconds:  {args.stuck_seconds}")
    print(f"[watch_bc] reset_key:      {args.reset_key}")
    print("[watch_bc] Focus/click Doom Retro during countdown.")
    print("[watch_bc] Ctrl+C stops and releases keys.")
    print("=" * 80)

    if args.state_json:
        print("[watch_bc] Stuck reset is enabled if the state JSON exists and contains x/y.")
    else:
        print("[watch_bc] No state JSON provided, so x/y stuck reset is disabled for now.")

    for i in range(5, 0, -1):
        print(f"[watch_bc] starting in {i}...")
        time.sleep(1)

    start_time = time.time()
    last_print_time = 0.0

    last_pos = None
    last_moved_time = time.time()

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[args.monitor]

            if (
                args.left is not None
                and args.top is not None
                and args.width is not None
                and args.height is not None
            ):
                monitor = {
                    "left": args.left,
                    "top": args.top,
                    "width": args.width,
                    "height": args.height,
                }

            while True:
                loop_start = time.time()

                if args.seconds > 0 and loop_start - start_time >= args.seconds:
                    break

                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)

                x = preprocess_image(img, image_width, image_height).to(device)

                with torch.no_grad():
                    keyboard_logits, mouse_pred, shoot_logits = model(x)

                    keyboard_id = int(torch.argmax(keyboard_logits, dim=1).item())
                    keyboard_label = id_to_label[keyboard_id]

                    mouse_xy = mouse_pred.squeeze(0).detach().cpu().numpy()
                    shoot_prob = torch.sigmoid(shoot_logits).item()

                wanted_keys = parse_keyboard_action(keyboard_label)

                dx = float(mouse_xy[0]) * mouse_scale * args.mouse_gain
                dy = float(mouse_xy[1]) * mouse_scale * args.mouse_gain

                do_shoot = shoot_prob >= args.shoot_threshold

                if not args.dry_run:
                    update_keys(keyboard, pressed_keys, wanted_keys)

                    # Avoid tiny jitter.
                    if abs(dx) >= 0.75 or abs(dy) >= 0.75:
                        mouse.move(int(round(dx)), int(round(dy)))

                    if do_shoot:
                        mouse.press(Button.left)
                    else:
                        mouse.release(Button.left)

                # Optional real x/y stuck reset.
                pos = read_player_position(args.state_json)
                if pos is not None:
                    if last_pos is None:
                        last_pos = pos
                        last_moved_time = loop_start
                    else:
                        dist = math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1])

                        if dist >= args.stuck_min_delta:
                            last_pos = pos
                            last_moved_time = loop_start

                        stuck_for = loop_start - last_moved_time

                        if stuck_for >= args.stuck_seconds:
                            print(
                                f"[watch_bc] STUCK: x/y barely changed for "
                                f"{stuck_for:.2f}s. Resetting with {args.reset_key}."
                            )

                            release_all_keys(keyboard, pressed_keys)
                            try:
                                mouse.release(Button.left)
                            except Exception:
                                pass

                            if not args.dry_run:
                                maybe_press_reset(keyboard, args.reset_key)

                            time.sleep(args.reset_wait)
                            last_pos = read_player_position(args.state_json)
                            last_moved_time = time.time()

                if loop_start - last_print_time >= 1.0:
                    print(
                        f"[watch_bc] action={keyboard_label:<28} "
                        f"keys={''.join(sorted(wanted_keys)) or '-':<4} "
                        f"mouse=({dx:+.1f},{dy:+.1f}) "
                        f"shoot={shoot_prob:.2f}"
                    )
                    last_print_time = loop_start

                elapsed = time.time() - loop_start
                sleep_time = frame_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[watch_bc] Ctrl+C received.")

    finally:
        release_all_keys(keyboard, pressed_keys)
        try:
            mouse.release(Button.left)
        except Exception:
            pass
        print("[watch_bc] stopped and released keys/buttons.")


if __name__ == "__main__":
    main()