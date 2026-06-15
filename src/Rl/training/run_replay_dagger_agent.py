from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time

import mss
import numpy as np
from PIL import Image

import torch
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_supervised_bc import DoomBCNet


DATASET = ROOT / "data" / "supervised_doomretro_v1"
DEFAULT_MODEL = DATASET / "models" / "supervised_bc_best.pt"
CACHE_DIR = DATASET / "replay_teacher_cache"
DEFAULT_FEATURES = CACHE_DIR / "teacher_features.npy"
DEFAULT_RECORDS = CACHE_DIR / "teacher_records.json"

DAGGER_DIR = DATASET / "dagger_rounds"


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

    id_to_label = {int(k): v for k, v in id_to_label.items()}

    model = DoomBCNet(num_keyboard_classes=len(label_to_id)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, label_to_id, id_to_label, config


def preprocess_model_image(img, width, height):
    img = img.convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def teacher_feature(img, width=32, height=18):
    img = img.convert("L")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).reshape(-1)
    return arr


def load_teacher_cache(features_path, records_path):
    if not Path(features_path).exists():
        raise SystemExit(f"Missing teacher features: {features_path}")

    if not Path(records_path).exists():
        raise SystemExit(f"Missing teacher records: {records_path}")

    features = np.load(features_path)
    records = json.loads(Path(records_path).read_text())

    if len(features) != len(records):
        raise SystemExit(
            f"Teacher cache mismatch: features={len(features)} records={len(records)}"
        )

    return features, records


def nearest_teacher(query_feature, teacher_features, teacher_records, chunk_size=4096):
    # Use int32/int64 math so squared pixel differences do not overflow.
    q = query_feature.astype(np.int32)

    best_idx = 0
    best_dist = None

    n = teacher_features.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        block = teacher_features[start:end].astype(np.int32)
        diff = block - q
        dists = np.sum(diff * diff, axis=1, dtype=np.int64)

        local_idx = int(np.argmin(dists))
        local_dist = int(dists[local_idx])

        if best_dist is None or local_dist < best_dist:
            best_dist = local_dist
            best_idx = start + local_idx

    return teacher_records[best_idx], best_dist


def is_teacher_movement_record(record):
    label = record.get("keyboard_action", "") or ""
    return (
        "move_forward" in label
        or "move_backward" in label
        or "strafe_left" in label
        or "strafe_right" in label
        or "use" in label
    )


def nearest_teacher_prefer_movement(query_feature, teacher_features, teacher_records, top_k=128):
    # First get distances to all teacher frames using safe int math.
    q = query_feature.astype(np.int32)
    block = teacher_features.astype(np.int32)
    diff = block - q
    dists = np.sum(diff * diff, axis=1, dtype=np.int64)

    # Look through the closest frames and prefer a non-no_op action.
    top_indices = np.argpartition(dists, min(top_k, len(dists) - 1))[:top_k]
    top_indices = sorted(top_indices, key=lambda i: int(dists[i]))

    best_any = int(top_indices[0])

    for idx in top_indices:
        record = teacher_records[int(idx)]
        if is_teacher_movement_record(record):
            return record, int(dists[int(idx)])

    return teacher_records[best_any], int(dists[best_any])


def to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_keyboard_action(label):
    if not label or label == "no_op":
        return set()

    keys = set()

    for part in label.split("+"):
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


def tap_key(keyboard, key, duration=0.035):
    keyboard.press(key)
    time.sleep(duration)
    keyboard.release(key)


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


def model_predict(model, img, device, id_to_label, config):
    image_width = int(config.get("image_width", 160))
    image_height = int(config.get("image_height", 90))
    mouse_scale = float(config.get("mouse_scale", 50.0))

    x = preprocess_model_image(img, image_width, image_height).to(device)

    with torch.no_grad():
        keyboard_logits, mouse_pred, shoot_logits = model(x)
        probs = torch.softmax(keyboard_logits, dim=1).squeeze(0)

        keyboard_id = int(torch.argmax(probs).item())
        keyboard_label = id_to_label[keyboard_id]
        keyboard_conf = float(probs[keyboard_id].item())

        entropy = float((-probs * torch.log(probs + 1e-8)).sum().item())

        mouse_xy = mouse_pred.squeeze(0).detach().cpu().numpy()
        shoot_prob = float(torch.sigmoid(shoot_logits).item())

    dx = float(mouse_xy[0]) * mouse_scale
    dy = float(mouse_xy[1]) * mouse_scale

    return {
        "keyboard_label": keyboard_label,
        "keyboard_conf": keyboard_conf,
        "entropy": entropy,
        "mouse_dx": dx,
        "mouse_dy": dy,
        "shoot_prob": shoot_prob,
    }


def teacher_action_from_record(record):
    buttons = str(record.get("mouse_buttons", "") or "")
    shoot = 1.0 if "left" in buttons.lower() else 0.0

    return {
        "keyboard_label": record.get("keyboard_action", "no_op") or "no_op",
        "keyboard_conf": 1.0,
        "entropy": 0.0,
        "mouse_dx": to_float(record.get("mouse_dx", 0.0)),
        "mouse_dy": to_float(record.get("mouse_dy", 0.0)),
        "shoot_prob": shoot,
    }


def is_movement_action(label):
    return (
        "move_forward" in label
        or "move_backward" in label
        or "strafe_left" in label
        or "strafe_right" in label
    )


def choose_action(
    model_action,
    teacher_action,
    args,
    state,
):
    label = model_action["keyboard_label"]
    source = "model"
    reason = "model_confident"

    if label == "no_op":
        state["consecutive_noop"] += 1
    else:
        state["consecutive_noop"] = 0

    if "use" in label:
        state["consecutive_use"] += 1
    else:
        state["consecutive_use"] = 0

    if args.teacher_always:
        return teacher_action, "teacher", "teacher_always"

    # Repeated bad behavior should override even confident predictions.
    # This is important because the model can be confidently wrong about no_op.
    if state["consecutive_noop"] >= args.teacher_after_noop:
        state["consecutive_noop"] = 0
        return teacher_action, "teacher", "repeated_noop"

    if state["consecutive_use"] >= args.teacher_after_use:
        state["consecutive_use"] = 0
        return teacher_action, "teacher", "repeated_use"

    if model_action["keyboard_conf"] < args.teacher_conf_threshold:
        return teacher_action, "teacher", f"low_conf_{model_action['keyboard_conf']:.2f}"

    return model_action, source, reason


def apply_runtime_filters(action, args, state, loop_time):
    label = action["keyboard_label"]

    if is_movement_action(label):
        state["last_movement_label"] = label
        state["last_movement_time"] = loop_time

    if label == "no_op":
        last_time = state.get("last_movement_time", 0.0)

        if last_time > 0.0 and loop_time - last_time <= args.movement_hold:
            label = state.get("last_movement_label", "move_forward")

    dx = action["mouse_dx"]
    dy = action["mouse_dy"]

    dx *= args.mouse_gain
    dy *= args.mouse_gain
    dy *= args.mouse_y_scale

    dx = max(-args.max_mouse_step, min(args.max_mouse_step, dx))
    dy = max(-args.max_mouse_step, min(args.max_mouse_step, dy))

    state["smooth_dx"] = args.mouse_smoothing * state["smooth_dx"] + (1.0 - args.mouse_smoothing) * dx
    state["smooth_dy"] = args.mouse_smoothing * state["smooth_dy"] + (1.0 - args.mouse_smoothing) * dy

    dx = state["smooth_dx"]
    dy = state["smooth_dy"]

    if abs(dx) < args.mouse_deadzone:
        dx = 0.0
        state["mouse_accum_x"] = 0.0

    if abs(dy) < args.mouse_deadzone:
        dy = 0.0
        state["mouse_accum_y"] = 0.0

    shoot_prob = action["shoot_prob"]

    return {
        "keyboard_label": label,
        "mouse_dx": dx,
        "mouse_dy": dy,
        "shoot_prob": shoot_prob,
    }


def prepare_dagger_writer(round_id):
    round_dir = DAGGER_DIR / f"round_{round_id:03d}"
    frames_dir = round_dir / "frames"
    actions_csv = round_dir / "actions.csv"

    frames_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame",
        "frame_path",
        "keyboard_action",
        "mouse_dx",
        "mouse_dy",
        "mouse_buttons",
        "source",
        "reason",
        "teacher_run",
        "teacher_frame_index",
        "teacher_distance",
        "model_action",
        "model_conf",
        "model_shoot",
    ]

    csv_file = actions_csv.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    return round_dir, frames_dir, actions_csv, csv_file, writer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--teacher-features", type=str, default=str(DEFAULT_FEATURES))
    parser.add_argument("--teacher-records", type=str, default=str(DEFAULT_RECORDS))

    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)

    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--round", type=int, default=1)

    parser.add_argument("--teacher-always", action="store_true")
    parser.add_argument("--teacher-conf-threshold", type=float, default=0.45)
    parser.add_argument("--teacher-after-noop", type=int, default=4)
    parser.add_argument("--teacher-after-use", type=int, default=6)

    parser.add_argument("--mouse-gain", type=float, default=0.45)
    parser.add_argument("--max-mouse-step", type=float, default=1.5)
    parser.add_argument("--mouse-smoothing", type=float, default=0.75)
    parser.add_argument("--mouse-deadzone", type=float, default=0.20)
    parser.add_argument("--mouse-y-scale", type=float, default=0.25)

    parser.add_argument("--shoot-threshold", type=float, default=0.40)
    parser.add_argument("--use-cooldown", type=float, default=0.40)
    parser.add_argument("--movement-hold", type=float, default=0.35)

    parser.add_argument("--reset-key", type=str, default="f9")
    parser.add_argument("--print-every", type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, label_to_id, id_to_label, config = load_model(Path(args.model), device)
    teacher_features, teacher_records = load_teacher_cache(
        args.teacher_features,
        args.teacher_records,
    )

    keyboard = KeyboardController()
    mouse = MouseController()
    pressed_keys = set()

    frame_dt = 1.0 / max(1.0, args.fps)

    dagger_csv_file = None
    dagger_writer = None
    dagger_frames_dir = None
    dagger_actions_csv = None

    if args.collect:
        round_dir, dagger_frames_dir, dagger_actions_csv, dagger_csv_file, dagger_writer = prepare_dagger_writer(args.round)

    state = {
        "consecutive_noop": 0,
        "consecutive_use": 0,
        "last_movement_label": "move_forward",
        "last_movement_time": 0.0,
        "smooth_dx": 0.0,
        "smooth_dy": 0.0,
        "mouse_accum_x": 0.0,
        "mouse_accum_y": 0.0,
        "last_use_time": 0.0,
    }

    print("=" * 80)
    print("[replay_dagger] Doom Retro Replay DAgger agent")
    print(f"[replay_dagger] model:          {args.model}")
    print(f"[replay_dagger] device:         {device}")
    print(f"[replay_dagger] teacher frames: {len(teacher_records)}")
    print(f"[replay_dagger] fps:            {args.fps}")
    print(f"[replay_dagger] dry_run:        {args.dry_run}")
    print(f"[replay_dagger] collect:        {args.collect}")
    print(f"[replay_dagger] teacher_always: {args.teacher_always}")
    print(f"[replay_dagger] conf threshold: {args.teacher_conf_threshold}")
    print(f"[replay_dagger] mouse gain:     {args.mouse_gain}")
    print(f"[replay_dagger] max mouse step: {args.max_mouse_step}")
    print(f"[replay_dagger] smoothing:      {args.mouse_smoothing}")
    print(f"[replay_dagger] deadzone:       {args.mouse_deadzone}")
    print("[replay_dagger] Click/focus Doom Retro during countdown.")
    print("[replay_dagger] Ctrl+C stops.")
    if args.collect:
        print(f"[replay_dagger] collecting to:   {dagger_actions_csv}")
    print("=" * 80)

    for i in range(5, 0, -1):
        print(f"[replay_dagger] starting in {i}...")
        time.sleep(1)

    start_time = time.time()
    last_print_time = 0.0
    frame_num = 0

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

                model_action = model_predict(model, img, device, id_to_label, config)

                q_feat = teacher_feature(img)
                # If the model is saying no_op, prefer a nearby teacher frame
                # with movement/use instead of copying another no_op frame.
                if model_action["keyboard_label"] == "no_op":
                    teacher_record, teacher_dist = nearest_teacher_prefer_movement(
                        q_feat,
                        teacher_features,
                        teacher_records,
                    )
                else:
                    teacher_record, teacher_dist = nearest_teacher(
                        q_feat,
                        teacher_features,
                        teacher_records,
                    )

                teacher_action = teacher_action_from_record(teacher_record)

                chosen_action, source, reason = choose_action(
                    model_action,
                    teacher_action,
                    args,
                    state,
                )

                filtered = apply_runtime_filters(
                    chosen_action,
                    args,
                    state,
                    loop_start,
                )

                keyboard_label = filtered["keyboard_label"]
                wanted_keys = parse_keyboard_action(keyboard_label)

                tap_use = "e" in wanted_keys
                if tap_use:
                    wanted_keys.discard("e")

                dx = filtered["mouse_dx"]
                dy = filtered["mouse_dy"]
                shoot_prob = filtered["shoot_prob"]
                do_shoot = shoot_prob >= args.shoot_threshold

                if not args.dry_run:
                    update_keys(keyboard, pressed_keys, wanted_keys)

                    now = time.time()
                    if tap_use and now - state["last_use_time"] >= args.use_cooldown:
                        tap_key(keyboard, "e")
                        state["last_use_time"] = now

                    state["mouse_accum_x"] += dx
                    state["mouse_accum_y"] += dy

                    move_x = int(round(state["mouse_accum_x"]))
                    move_y = int(round(state["mouse_accum_y"]))

                    if move_x != 0 or move_y != 0:
                        mouse.move(move_x, move_y)
                        state["mouse_accum_x"] -= move_x
                        state["mouse_accum_y"] -= move_y

                    if do_shoot:
                        mouse.press(Button.left)
                    else:
                        mouse.release(Button.left)

                if args.collect and source == "teacher":
                    frame_name = f"frame_{frame_num:06d}.png"
                    frame_path = dagger_frames_dir / frame_name
                    img.save(frame_path)

                    mouse_buttons = "left" if teacher_action["shoot_prob"] >= 0.5 else ""

                    dagger_writer.writerow({
                        "frame": frame_name,
                        "frame_path": str(frame_path),
                        "keyboard_action": teacher_action["keyboard_label"],
                        "mouse_dx": teacher_action["mouse_dx"],
                        "mouse_dy": teacher_action["mouse_dy"],
                        "mouse_buttons": mouse_buttons,
                        "source": source,
                        "reason": reason,
                        "teacher_run": teacher_record.get("run_name", ""),
                        "teacher_frame_index": teacher_record.get("frame_index", ""),
                        "teacher_distance": teacher_dist,
                        "model_action": model_action["keyboard_label"],
                        "model_conf": f"{model_action['keyboard_conf']:.4f}",
                        "model_shoot": f"{model_action['shoot_prob']:.4f}",
                    })

                if loop_start - last_print_time >= args.print_every:
                    print(
                        f"[replay_dagger] src={source:<7} reason={reason:<16} "
                        f"act={keyboard_label:<28} keys={''.join(sorted(wanted_keys)) or '-':<4} "
                        f"mouse=({dx:+.2f},{dy:+.2f}) "
                        f"shoot={shoot_prob:.2f} "
                        f"model={model_action['keyboard_label']:<22} "
                        f"conf={model_action['keyboard_conf']:.2f} "
                        f"teacher={teacher_action['keyboard_label']:<22} "
                        f"dist={teacher_dist}"
                    )
                    last_print_time = loop_start

                frame_num += 1

                elapsed = time.time() - loop_start
                sleep_time = frame_dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[replay_dagger] Ctrl+C received.")

    finally:
        release_all_keys(keyboard, pressed_keys)

        try:
            mouse.release(Button.left)
        except Exception:
            pass

        if dagger_csv_file is not None:
            dagger_csv_file.close()

        print("[replay_dagger] stopped and released keys/buttons.")

        if args.collect:
            print(f"[replay_dagger] saved DAgger round actions: {dagger_actions_csv}")


if __name__ == "__main__":
    main()