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



def read_advice_json(path):
    if not path:
        return None

    try:
        import json
        import time
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return None

        # Avoid using stale advice if the watcher stopped.
        if time.time() - p.stat().st_mtime > 2.0:
            return None

        return json.loads(p.read_text())
    except Exception:
        return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
    parser.add_argument("--advice-json", type=str, default="")
    parser.add_argument("--advice-mode", type=str, default="soft",
                        choices=["off", "soft", "override"])
    parser.add_argument("--advice-turn-gain", type=float, default=0.020)
    parser.add_argument("--advice-max-turn", type=float, default=0.25)
    parser.add_argument("--advice-invert-turn", action="store_true")
    parser.add_argument("--advice-final-use", action="store_true")
    parser.add_argument("--advice-stop-turn-deg", type=float, default=35.0)
    parser.add_argument("--advice-no-shoot-offroute", action="store_true")
    parser.add_argument("--advice-forward-on-route", action="store_true")
    parser.add_argument("--stuck-seconds", type=float, default=3.0)
    parser.add_argument("--stuck-min-delta", type=float, default=1.0)
    parser.add_argument("--reset-key", type=str, default="f9")
    parser.add_argument("--reset-wait", type=float, default=1.5)

    # If the model keeps choosing no_op, this lets the best movement action win
    # when its probability is close to no_op.
    parser.add_argument("--anti-noop", action="store_true")
    parser.add_argument("--noop-margin", type=float, default=0.15)

    # Doom "use" should be tapped, not held every frame.
    parser.add_argument("--use-cooldown", type=float, default=0.35)
    parser.add_argument("--max-consecutive-use", type=int, default=8)
    parser.add_argument("--use-spam-fallback", type=str, default="move_forward")

    # Runtime safety guards for the first behavior-cloned test agent.
    parser.add_argument("--max-consecutive-noop", type=int, default=6)
    parser.add_argument("--noop-fallback", type=str, default="move_forward")
    parser.add_argument("--movement-hold", type=float, default=0.25)

    # Smooth and clamp mouse output so one bad prediction cannot hard-flick.
    parser.add_argument("--max-mouse-step", type=float, default=2.0)
    parser.add_argument("--mouse-smoothing", type=float, default=0.65)
    parser.add_argument("--no-mouse-on-noop", action="store_true")

    # Ignore tiny mouse drift so the camera does not constantly rotate.
    parser.add_argument("--mouse-deadzone", type=float, default=0.35)
    parser.add_argument("--mouse-y-scale", type=float, default=0.35)

    # Anti-spin guard: prevents tiny repeated same-direction mouse drift
    # from becoming a constant circle while the agent holds move_forward.
    parser.add_argument("--anti-spin", action="store_true")
    parser.add_argument("--anti-spin-min-dx", type=float, default=0.08)
    parser.add_argument("--anti-spin-max-same", type=int, default=8)
    parser.add_argument("--anti-spin-dampen", type=float, default=0.25)
    parser.add_argument("--turn-rate-limit", type=float, default=0.0)

    parser.add_argument("--auto-mouse-fps-scale", action="store_true")
    parser.add_argument("--mouse-reference-fps", type=float, default=20.0)
    parser.add_argument("--mouse-scale-min", type=float, default=0.08)
    parser.add_argument("--mouse-scale-max", type=float, default=1.25)
    parser.add_argument("--mouse-scale-smoothing", type=float, default=0.90)

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
    print("[supervised_agent] Supervised Doom Retro agent")
    print(f"[supervised_agent] model:          {args.model}")
    print(f"[supervised_agent] device:         {device}")
    print(f"[supervised_agent] fps:            {args.fps}")
    print(f"[supervised_agent] image size:     {image_width}x{image_height}")
    print(f"[supervised_agent] mouse_scale:    {mouse_scale}")
    print(f"[supervised_agent] mouse_gain:     {args.mouse_gain}")
    print(f"[supervised_agent] dry_run:        {args.dry_run}")
    print(f"[supervised_agent] state_json:     {args.state_json or 'none'}")
    print(f"[supervised_agent] stuck_seconds:  {args.stuck_seconds}")
    print(f"[supervised_agent] reset_key:      {args.reset_key}")
    print("[supervised_agent] Focus/click Doom Retro during countdown.")
    print("[supervised_agent] Ctrl+C stops and releases keys.")
    print("=" * 80)

    if args.state_json:
        print("[supervised_agent] Stuck reset is enabled if the state JSON exists and contains x/y.")
    else:
        print("[supervised_agent] No state JSON provided, so x/y stuck reset is disabled for now.")

    for i in range(5, 0, -1):
        print(f"[supervised_agent] starting in {i}...")
        time.sleep(1)

    start_time = time.time()
    last_print_time = 0.0

    last_pos = None
    last_moved_time = time.time()

    last_use_time = 0.0
    consecutive_use_count = 0

    consecutive_noop_count = 0
    last_movement_label = "move_forward"
    last_movement_time = 0.0

    smooth_dx = 0.0
    smooth_dy = 0.0

    mouse_accum_x = 0.0
    mouse_accum_y = 0.0

    same_turn_sign = 0
    same_turn_frames = 0
    turn_rate_window_start = time.time()
    turn_rate_accum = 0.0

    last_step_time = time.time()
    auto_mouse_scale = 1.0

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

                dt = max(1e-6, loop_start - last_step_time)
                last_step_time = loop_start
                measured_fps = 1.0 / dt
                raw_auto_scale = args.mouse_reference_fps / max(1.0, measured_fps)
                raw_auto_scale = max(args.mouse_scale_min, min(args.mouse_scale_max, raw_auto_scale))
                auto_mouse_scale = (
                    args.mouse_scale_smoothing * auto_mouse_scale
                    + (1.0 - args.mouse_scale_smoothing) * raw_auto_scale
                )

                if args.seconds > 0 and loop_start - start_time >= args.seconds:
                    break

                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)

                x = preprocess_image(img, image_width, image_height).to(device)

                with torch.no_grad():
                    keyboard_logits, mouse_pred, shoot_logits = model(x)

                    keyboard_probs = torch.softmax(keyboard_logits, dim=1).squeeze(0)
                    keyboard_id = int(torch.argmax(keyboard_probs).item())
                    keyboard_label = id_to_label[keyboard_id]

                    if args.anti_noop and keyboard_label == "no_op":
                        noop_prob = float(keyboard_probs[keyboard_id].item())

                        movement_candidates = []
                        for idx, label in id_to_label.items():
                            if label != "no_op":
                                prob = float(keyboard_probs[int(idx)].item())
                                movement_candidates.append((prob, int(idx), label))

                        movement_candidates.sort(reverse=True)

                        if movement_candidates:
                            best_move_prob, best_move_id, best_move_label = movement_candidates[0]

                            # If no_op is not much more confident than movement, move.
                            if best_move_prob >= noop_prob - args.noop_margin:
                                keyboard_id = best_move_id
                                keyboard_label = best_move_label

                    mouse_xy = mouse_pred.squeeze(0).detach().cpu().numpy()
                    shoot_prob = torch.sigmoid(shoot_logits).item()

                raw_keyboard_label = keyboard_label

                movement_like = (
                    "move_forward" in keyboard_label
                    or "move_backward" in keyboard_label
                    or "strafe_left" in keyboard_label
                    or "strafe_right" in keyboard_label
                )

                if keyboard_label == "no_op":
                    consecutive_noop_count += 1
                else:
                    consecutive_noop_count = 0

                if movement_like:
                    last_movement_label = keyboard_label
                    last_movement_time = loop_start

                # Do not let one or two no_op frames instantly stop movement.
                if keyboard_label == "no_op":
                    if last_movement_time > 0 and (loop_start - last_movement_time) <= args.movement_hold:
                        keyboard_label = last_movement_label
                    elif consecutive_noop_count >= args.max_consecutive_noop:
                        keyboard_label = args.noop_fallback
                        consecutive_noop_count = 0

                if "use" in keyboard_label:
                    consecutive_use_count += 1
                else:
                    consecutive_use_count = 0

                if consecutive_use_count > args.max_consecutive_use:
                    keyboard_label = args.use_spam_fallback

                wanted_keys = parse_keyboard_action(keyboard_label)

                # Treat E/use as a tap, not as a held key.
                tap_use = "e" in wanted_keys
                if tap_use:
                    wanted_keys.discard("e")

                raw_dx = float(mouse_xy[0]) * mouse_scale * args.mouse_gain
                raw_dy = float(mouse_xy[1]) * mouse_scale * args.mouse_gain

                if args.auto_mouse_fps_scale:
                    raw_dx *= auto_mouse_scale
                    raw_dy *= auto_mouse_scale

                # Clamp hard flicks.
                raw_dx = max(-args.max_mouse_step, min(args.max_mouse_step, raw_dx))
                raw_dy = max(-args.max_mouse_step, min(args.max_mouse_step, raw_dy))

                # Smooth mouse over time.
                smooth_dx = args.mouse_smoothing * smooth_dx + (1.0 - args.mouse_smoothing) * raw_dx
                smooth_dy = args.mouse_smoothing * smooth_dy + (1.0 - args.mouse_smoothing) * raw_dy

                dx = smooth_dx
                dy = smooth_dy * args.mouse_y_scale

                # Deadzone: ignore tiny constant drift.
                if abs(dx) < args.mouse_deadzone:
                    dx = 0.0
                    mouse_accum_x = 0.0

                if abs(dy) < args.mouse_deadzone:
                    dy = 0.0
                    mouse_accum_y = 0.0

                if args.no_mouse_on_noop and raw_keyboard_label == "no_op":
                    dx = 0.0
                    dy = 0.0
                    mouse_accum_x = 0.0
                    mouse_accum_y = 0.0

                if args.anti_spin:
                    # Track repeated same-direction turning while moving.
                    if dx > args.anti_spin_min_dx:
                        turn_sign = 1
                    elif dx < -args.anti_spin_min_dx:
                        turn_sign = -1
                    else:
                        turn_sign = 0

                    if movement_like and turn_sign != 0:
                        if turn_sign == same_turn_sign:
                            same_turn_frames += 1
                        else:
                            same_turn_sign = turn_sign
                            same_turn_frames = 1
                    else:
                        same_turn_sign = 0
                        same_turn_frames = 0

                    # If it keeps turning the same direction for too long,
                    # damp both current dx and accumulated fractional dx.
                    if same_turn_frames > args.anti_spin_max_same:
                        dx *= args.anti_spin_dampen
                        mouse_accum_x *= args.anti_spin_dampen

                    # Optional per-second turn budget.
                    if args.turn_rate_limit > 0:
                        now_rate = time.time()
                        if now_rate - turn_rate_window_start >= 1.0:
                            turn_rate_window_start = now_rate
                            turn_rate_accum = 0.0

                        proposed = turn_rate_accum + dx
                        if abs(proposed) > args.turn_rate_limit:
                            remaining = max(0.0, args.turn_rate_limit - abs(turn_rate_accum))
                            dx = math.copysign(remaining, dx) if dx != 0 else 0.0

                        turn_rate_accum += dx

                advice_label = "none"
                advice_idle = False

                if args.advice_mode != "off":
                    advice = read_advice_json(args.advice_json)
                else:
                    advice = None

                if advice:
                    route = advice.get("route", {})
                    stuck_info = advice.get("stuck", {})

                    route_status = route.get("status", "")
                    route_advice = route.get("advice", "")
                    turn_error = float(route.get("turn_error", 0.0))

                    advice_idle = bool(stuck_info.get("idle_or_paused", False))
                    advice_stuck = bool(stuck_info.get("stuck", False)) and not advice_idle

                    advice_label = f"{route_status}:{route_advice}"

                    # Convert route turn error into mouse dx.
                    # Positive route error means "turn left" in the watcher.
                    # For normal Doom mouse, negative dx usually turns left.
                    turn_dx = -turn_error * args.advice_turn_gain
                    if args.advice_invert_turn:
                        turn_dx = -turn_dx
                    turn_dx = clamp(turn_dx, -args.advice_max_turn, args.advice_max_turn)

                    should_route_correct = route_advice in ("turn_left", "turn_right")
                    should_override = (
                        args.advice_mode == "override"
                        or route_status == "OFF_ROUTE"
                        or advice_stuck
                    )

                    if route_advice == "finish_or_use":
                        if args.advice_final_use:
                            tap_use = True
                        # Stop forcing forward once we are at the final route zone.
                        if should_override:
                            wanted_keys = set()

                    elif should_route_correct:
                        if should_override:
                            dx = turn_dx
                            wanted_keys = set(wanted_keys)

                            # If the route error is large, rotate first.
                            # Do not keep charging forward into walls.
                            if abs(turn_error) >= args.advice_stop_turn_deg:
                                wanted_keys.discard("w")
                                wanted_keys.discard("a")
                                wanted_keys.discard("d")
                            else:
                                wanted_keys.add("w")
                        else:
                            # Soft blend: keep BC/model behavior, add route correction.
                            dx = (0.70 * dx) + (0.30 * turn_dx)
                            if args.advice_forward_on_route:
                                wanted_keys = set(wanted_keys)
                                wanted_keys.add("w")

                    elif route_advice == "move_forward" and args.advice_forward_on_route:
                        wanted_keys = set(wanted_keys)
                        wanted_keys.add("w")

                    if args.advice_no_shoot_offroute and route_status == "OFF_ROUTE":
                        shoot_prob = 0.0

                do_shoot = shoot_prob >= args.shoot_threshold

                if not args.dry_run:
                    update_keys(keyboard, pressed_keys, wanted_keys)

                    now = time.time()
                    if tap_use and now - last_use_time >= args.use_cooldown:
                        keyboard.press("e")
                        time.sleep(0.035)
                        keyboard.release("e")
                        last_use_time = now

                    # Accumulate fractional mouse movement so small aim corrections
                    # are not thrown away every frame.
                    mouse_accum_x += dx
                    mouse_accum_y += dy

                    move_x = int(round(mouse_accum_x))
                    move_y = int(round(mouse_accum_y))

                    if move_x != 0 or move_y != 0:
                        mouse.move(move_x, move_y)
                        mouse_accum_x -= move_x
                        mouse_accum_y -= move_y

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

                        if stuck_for >= args.stuck_seconds and not advice_idle:
                            print(
                                f"[supervised_agent] STUCK: x/y barely changed for "
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
                        f"[supervised_agent] action={keyboard_label:<28} "
                        f"keys={''.join(sorted(wanted_keys)) or '-':<4} "
                        f"mouse=({dx:+.1f},{dy:+.1f}) "
                        f"shoot={shoot_prob:.2f} "
                        f"advice={advice_label}"
                    )
                    last_print_time = loop_start

                elapsed = time.time() - loop_start
                sleep_time = frame_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[supervised_agent] Ctrl+C received.")

    finally:
        release_all_keys(keyboard, pressed_keys)
        try:
            mouse.release(Button.left)
        except Exception:
            pass
        print("[supervised_agent] stopped and released keys/buttons.")


if __name__ == "__main__":
    main()