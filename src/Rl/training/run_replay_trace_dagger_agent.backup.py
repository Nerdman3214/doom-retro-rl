from pathlib import Path
import argparse
import re
import sys
import time

import mss
import numpy as np
from PIL import Image

import torch
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.run_replay_dagger_agent as base


DEFAULT_MODEL = base.DEFAULT_MODEL
DEFAULT_FEATURES = base.DEFAULT_FEATURES
DEFAULT_RECORDS = base.DEFAULT_RECORDS


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def frame_number_from_record(record):
    value = record.get("frame_index", "")

    try:
        return int(value)
    except Exception:
        pass

    frame_path = str(record.get("frame_path", ""))
    match = re.search(r"frame_(\d+)", frame_path)

    if match:
        return int(match.group(1))

    return safe_int(record.get("teacher_index", 0), 0)


def build_teacher_run_index(records):
    runs = {}

    for idx, record in enumerate(records):
        run_name = record.get("run_name", "") or "unknown_run"
        record["_cache_index"] = idx
        record["_frame_num"] = frame_number_from_record(record)

        runs.setdefault(run_name, []).append(record)

    for run_name, seq in runs.items():
        seq.sort(key=lambda r: r.get("_frame_num", 0))

        for pos, record in enumerate(seq):
            record["_run_pos"] = pos

    return runs


def is_movement_label(label):
    label = label or ""

    return (
        "move_forward" in label
        or "move_backward" in label
        or "strafe_left" in label
        or "strafe_right" in label
        or "use" in label
    )


def action_description(record):
    label = record.get("keyboard_action", "") or "no_op"
    dx = base.to_float(record.get("mouse_dx", 0.0))
    dy = base.to_float(record.get("mouse_dy", 0.0))
    buttons = str(record.get("mouse_buttons", "") or "").lower()

    parts = []

    if label and label != "no_op":
        parts.append(label)
    else:
        parts.append("no_op")

    if dx > 1.0:
        parts.append("turn_right")
    elif dx < -1.0:
        parts.append("turn_left")

    if dy > 1.0:
        parts.append("look_down")
    elif dy < -1.0:
        parts.append("look_up")

    if "left" in buttons:
        parts.append("shoot")

    return "+".join(parts)


def describe_trace(segment, fps, max_lines=10):
    if not segment:
        return []

    grouped = []
    current_desc = action_description(segment[0])
    count = 0

    for record in segment:
        desc = action_description(record)

        if desc == current_desc:
            count += 1
        else:
            grouped.append((current_desc, count))
            current_desc = desc
            count = 1

    grouped.append((current_desc, count))

    lines = []

    for idx, (desc, frames) in enumerate(grouped[:max_lines], start=1):
        seconds = frames / max(1.0, fps)
        lines.append(f"{idx:02d}. {desc:<40} for {seconds:.2f}s")

    if len(grouped) > max_lines:
        lines.append(f"... plus {len(grouped) - max_lines} more action chunks")

    return lines


def get_trace_segment(record, run_index, fps, before_seconds, after_seconds):
    run_name = record.get("run_name", "") or "unknown_run"
    seq = run_index.get(run_name)

    if not seq:
        return []

    pos = record.get("_run_pos")

    if pos is None:
        cache_index = record.get("_cache_index")

        for i, candidate in enumerate(seq):
            if candidate.get("_cache_index") == cache_index:
                pos = i
                break

    if pos is None:
        pos = 0

    before_frames = int(before_seconds * fps)
    after_frames = int(after_seconds * fps)

    start = max(0, int(pos) - before_frames)
    end = min(len(seq), int(pos) + after_frames + 1)

    return seq[start:end]


def trim_leading_trace_noop(segment, min_action_frames=4):
    if not segment:
        return segment

    for i, record in enumerate(segment):
        label = record.get("keyboard_action", "") or "no_op"
        buttons = str(record.get("mouse_buttons", "") or "").lower()
        dx = abs(base.to_float(record.get("mouse_dx", 0.0)))
        dy = abs(base.to_float(record.get("mouse_dy", 0.0)))

        useful = (
            label != "no_op"
            or "left" in buttons
            or dx >= 1.0
            or dy >= 1.0
        )

        if useful:
            start = max(0, i - min_action_frames)
            return segment[start:]

    return segment


def nearest_teacher_for_trace(query_feature, teacher_features, teacher_records, model_label):
    prefer_movement = getattr(
        base,
        "nearest_teacher_prefer_movement",
        base.nearest_teacher,
    )

    if model_label == "no_op":
        return prefer_movement(query_feature, teacher_features, teacher_records)

    return base.nearest_teacher(query_feature, teacher_features, teacher_records)


def should_start_trace(reason, source, args):
    if not args.trace_recovery:
        return False

    if source != "teacher":
        return False

    if reason in {"repeated_noop", "repeated_use"}:
        return True

    if args.trace_on_low_conf and reason.startswith("low_conf"):
        return True

    return False


def pop_trace_action(state):
    if not state["trace_queue"]:
        return None

    record = state["trace_queue"].pop(0)
    return record


def runtime_apply_inputs(
    args,
    keyboard,
    mouse,
    pressed_keys,
    state,
    filtered,
):
    keyboard_label = filtered["keyboard_label"]
    wanted_keys = base.parse_keyboard_action(keyboard_label)

    tap_use = "e" in wanted_keys

    if tap_use:
        wanted_keys.discard("e")

    dx = filtered["mouse_dx"]
    dy = filtered["mouse_dy"]
    shoot_prob = filtered["shoot_prob"]
    do_shoot = shoot_prob >= args.shoot_threshold

    if do_shoot:
        state["shoot_streak"] = state.get("shoot_streak", 0) + 1
    else:
        state["shoot_streak"] = 0

    if state["shoot_streak"] > args.max_shoot_streak:
        do_shoot = False
        shoot_prob = 0.0

    if not args.dry_run:
        base.update_keys(keyboard, pressed_keys, wanted_keys)

        now = time.time()

        if tap_use and now - state["last_use_time"] >= args.use_cooldown:
            base.tap_key(keyboard, "e")
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

    return wanted_keys, dx, dy, shoot_prob


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

    parser.add_argument("--teacher-conf-threshold", type=float, default=0.60)
    parser.add_argument("--teacher-after-noop", type=int, default=3)
    parser.add_argument("--teacher-after-use", type=int, default=5)
    parser.add_argument("--teacher-always", action="store_true")

    parser.add_argument("--trace-recovery", action="store_true")
    parser.add_argument("--trace-on-low-conf", action="store_true")
    parser.add_argument("--trace-before", type=float, default=0.75)
    parser.add_argument("--trace-after", type=float, default=3.50)
    parser.add_argument("--trace-cooldown", type=float, default=2.0)
    parser.add_argument("--describe-trace", action="store_true")

    # Keep trace replay useful instead of replaying teacher idle/spin at the start.
    parser.add_argument("--skip-leading-trace-noop", action="store_true")
    parser.add_argument("--trace-min-action-frames", type=int, default=4)

    # Prevent ammo waste during trace replay.
    parser.add_argument("--max-shoot-streak", type=int, default=12)

    parser.add_argument("--mouse-gain", type=float, default=0.45)
    parser.add_argument("--max-mouse-step", type=float, default=1.2)
    parser.add_argument("--mouse-smoothing", type=float, default=0.75)
    parser.add_argument("--mouse-deadzone", type=float, default=0.20)
    parser.add_argument("--mouse-y-scale", type=float, default=0.25)

    parser.add_argument("--shoot-threshold", type=float, default=0.40)
    parser.add_argument("--use-cooldown", type=float, default=0.40)
    parser.add_argument("--movement-hold", type=float, default=0.35)
    parser.add_argument("--print-every", type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, label_to_id, id_to_label, config = base.load_model(Path(args.model), device)

    teacher_features, teacher_records = base.load_teacher_cache(
        args.teacher_features,
        args.teacher_records,
    )

    run_index = build_teacher_run_index(teacher_records)

    keyboard = KeyboardController()
    mouse = MouseController()
    pressed_keys = set()

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
        "trace_queue": [],
        "last_trace_time": 0.0,
        "trace_id": 0,
        "shoot_streak": 0,
    }

    frame_dt = 1.0 / max(1.0, args.fps)

    print("=" * 80)
    print("[trace_dagger] Doom Retro Replay-Trace DAgger agent")
    print(f"[trace_dagger] model:             {args.model}")
    print(f"[trace_dagger] device:            {device}")
    print(f"[trace_dagger] teacher frames:    {len(teacher_records)}")
    print(f"[trace_dagger] teacher runs:      {len(run_index)}")
    print(f"[trace_dagger] fps:               {args.fps}")
    print(f"[trace_dagger] dry_run:           {args.dry_run}")
    print(f"[trace_dagger] trace_recovery:    {args.trace_recovery}")
    print(f"[trace_dagger] trace_before:      {args.trace_before}")
    print(f"[trace_dagger] trace_after:       {args.trace_after}")
    print(f"[trace_dagger] teacher threshold: {args.teacher_conf_threshold}")
    print("[trace_dagger] Click/focus Doom Retro during countdown.")
    print("[trace_dagger] Ctrl+C stops.")
    print("=" * 80)

    for i in range(5, 0, -1):
        print(f"[trace_dagger] starting in {i}...")
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

                model_action = base.model_predict(
                    model,
                    img,
                    device,
                    id_to_label,
                    config,
                )

                q_feat = base.teacher_feature(img)

                trace_record = pop_trace_action(state)

                if trace_record is not None:
                    chosen_action = base.teacher_action_from_record(trace_record)
                    source = "trace"
                    reason = f"trace_replay_{len(state['trace_queue'])}"
                    teacher_record = trace_record
                    teacher_dist = -1

                else:
                    teacher_record, teacher_dist = nearest_teacher_for_trace(
                        q_feat,
                        teacher_features,
                        teacher_records,
                        model_action["keyboard_label"],
                    )

                    teacher_action = base.teacher_action_from_record(teacher_record)

                    chosen_action, source, reason = base.choose_action(
                        model_action,
                        teacher_action,
                        args,
                        state,
                    )

                    can_trace = loop_start - state["last_trace_time"] >= args.trace_cooldown

                    if can_trace and should_start_trace(reason, source, args):
                        segment = get_trace_segment(
                            teacher_record,
                            run_index,
                            args.fps,
                            args.trace_before,
                            args.trace_after,
                        )

                        movement_segment = [
                            r for r in segment
                            if is_movement_label(r.get("keyboard_action", ""))
                        ]

                        if movement_segment:
                            if args.skip_leading_trace_noop:
                                segment = trim_leading_trace_noop(
                                    segment,
                                    min_action_frames=args.trace_min_action_frames,
                                )

                            state["trace_queue"] = segment
                            state["last_trace_time"] = loop_start
                            state["trace_id"] += 1

                            trace_record = pop_trace_action(state)
                            chosen_action = base.teacher_action_from_record(trace_record)
                            source = "trace"
                            reason = f"trace_start_{reason}"
                            teacher_record = trace_record

                            if args.describe_trace:
                                run_name = trace_record.get("run_name", "unknown_run")
                                frame_num_match = trace_record.get("_frame_num", "?")
                                print("-" * 80)
                                print(
                                    f"[trace_dagger] TRACE #{state['trace_id']} "
                                    f"matched {run_name} near frame {frame_num_match}"
                                )
                                print("[trace_dagger] teacher summary:")

                                for line in describe_trace(segment, args.fps):
                                    print(f"[trace_dagger]   {line}")

                                print("-" * 80)

                filtered = base.apply_runtime_filters(
                    chosen_action,
                    args,
                    state,
                    loop_start,
                )

                wanted_keys, dx, dy, shoot_prob = runtime_apply_inputs(
                    args,
                    keyboard,
                    mouse,
                    pressed_keys,
                    state,
                    filtered,
                )

                if loop_start - last_print_time >= args.print_every:
                    print(
                        f"[trace_dagger] src={source:<7} reason={reason:<22} "
                        f"act={filtered['keyboard_label']:<28} "
                        f"keys={''.join(sorted(wanted_keys)) or '-':<4} "
                        f"mouse=({dx:+.2f},{dy:+.2f}) "
                        f"shoot={shoot_prob:.2f} "
                        f"model={model_action['keyboard_label']:<22} "
                        f"conf={model_action['keyboard_conf']:.2f} "
                        f"teacher={teacher_record.get('keyboard_action', 'no_op'):<22} "
                        f"dist={teacher_dist}"
                    )

                    last_print_time = loop_start

                frame_num += 1

                elapsed = time.time() - loop_start
                sleep_time = frame_dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[trace_dagger] Ctrl+C received.")

    finally:
        base.release_all_keys(keyboard, pressed_keys)

        try:
            mouse.release(Button.left)
        except Exception:
            pass

        print("[trace_dagger] stopped and released keys/buttons.")


if __name__ == "__main__":
    main()