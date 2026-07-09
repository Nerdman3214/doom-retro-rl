from pathlib import Path
import argparse
import json
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


def trace_key(record, bucket_size=20):
    run_name = record.get("run_name", "unknown_run")
    frame_num = safe_int(record.get("_frame_num", 0), 0)
    return f"{run_name}:{frame_num // max(1, bucket_size)}"


def screen_delta(a, b):
    if a is None or b is None:
        return 999999.0

    aa = a.astype(np.int16)
    bb = b.astype(np.int16)
    return float(np.mean(np.abs(aa - bb)))


def make_escape_probe(state):
    # Short reversible actions. Not random spinning; just unstuck probes.
    state["probe_queue"] = [
        {
            "keyboard_label": "move_backward",
            "keyboard_conf": 1.0,
            "entropy": 0.0,
            "mouse_dx": 0.0,
            "mouse_dy": 0.0,
            "shoot_prob": 0.0,
        },
        {
            "keyboard_label": "strafe_left",
            "keyboard_conf": 1.0,
            "entropy": 0.0,
            "mouse_dx": -1.0,
            "mouse_dy": 0.0,
            "shoot_prob": 0.0,
        },
        {
            "keyboard_label": "move_forward+strafe_right",
            "keyboard_conf": 1.0,
            "entropy": 0.0,
            "mouse_dx": 0.8,
            "mouse_dy": 0.0,
            "shoot_prob": 0.0,
        },
        {
            "keyboard_label": "use",
            "keyboard_conf": 1.0,
            "entropy": 0.0,
            "mouse_dx": 0.0,
            "mouse_dy": 0.0,
            "shoot_prob": 0.0,
        },
    ]


def maybe_auto_reset(state, args, keyboard, mouse, pressed_keys, now):
    if not args.auto_reset:
        return False

    if now - state.get("last_reset_time", 0.0) < args.reset_cooldown:
        return False

    print("[auto_reset] stuck/died detected; pressing reset")
    base.release_all_keys(keyboard, pressed_keys)

    try:
        mouse.release(Button.left)
    except Exception:
        pass

    base.maybe_press_reset(keyboard, args.reset_key)

    state["last_reset_time"] = now
    state["reset_count"] = state.get("reset_count", 0) + 1
    state["consecutive_stuck"] = 0
    state["trace_queue"] = []
    state["probe_queue"] = []
    state["recent_traces"] = {}
    state["last_trace_time"] = 0.0
    state["last_screen_feature"] = None
    state["last_screen_change_time"] = now
    state["last_position"] = None
    state["last_position_change_time"] = now
    state["mouse_spin_frames"] = 0
    state["last_mouse_sign"] = 0
    state["mouse_spin_until"] = 0.0
    state["smooth_dx"] = 0.0
    state["smooth_dy"] = 0.0
    state["mouse_accum_x"] = 0.0
    state["mouse_accum_y"] = 0.0

    if args.reset_sleep > 0.0:
        time.sleep(args.reset_sleep)

    return True


def pop_probe_action(state):
    queue = state.get("probe_queue", [])

    if not queue:
        return None

    return queue.pop(0)


def register_trace_or_block(state, args, record, now):
    if not args.guardrails:
        return False

    # Use configurable bucketing for frame deduplication.
    key = trace_key(record, bucket_size=getattr(args, "trace_bucket_size", 20))
    recent = state.get("recent_traces", {})

    # Remove old trace entries.
    recent = {
        k: v for k, v in recent.items()
        if now - v["time"] <= args.trace_repeat_window
    }

    entry = recent.get(key, {"count": 0, "time": now})
    entry["count"] += 1
    entry["time"] = now
    recent[key] = entry
    state["recent_traces"] = recent

    # Allow forcing trace replay (ignore repeated-trace guardrail) when the
    # user explicitly requests it via CLI. This is useful when you want the
    # agent to faithfully replay teacher traces even if they repeat.
    if getattr(args, "trace_force", False):
        return False

    if entry["count"] > args.max_same_trace:
        print(f"[guardrail] blocked repeated trace {key}; using escape probe")
        make_escape_probe(state)
        return True

    return False


def detect_screen_stuck(state, args, q_feat, final_label, now):
    if not args.guardrails:
        return False

    delta = screen_delta(q_feat, state.get("last_screen_feature"))

    if state.get("last_screen_feature") is None:
        state["last_screen_feature"] = q_feat.copy()
        state["last_screen_change_time"] = now
        return False

    if delta >= args.screen_motion_delta:
        state["last_screen_feature"] = q_feat.copy()
        state["last_screen_change_time"] = now
        return False

    stuck_time = now - state.get("last_screen_change_time", now)

    moving_forward = (
        final_label == "move_forward"
        or final_label == "move_forward+strafe_left"
        or final_label == "move_forward+strafe_right"
    )

    return moving_forward and stuck_time >= args.screen_stuck_seconds


def apply_mouse_spin_guard(state, args, filtered, now):
    if not args.guardrails:
        return filtered

    dx = filtered["mouse_dx"]

    if abs(dx) < args.mouse_spin_threshold:
        state["mouse_spin_frames"] = 0
        state["last_mouse_sign"] = 0
        return filtered

    sign = 1 if dx > 0 else -1

    if sign == state.get("last_mouse_sign", 0):
        state["mouse_spin_frames"] = state.get("mouse_spin_frames", 0) + 1
    else:
        state["mouse_spin_frames"] = 1
        state["last_mouse_sign"] = sign

    if state["mouse_spin_frames"] >= args.mouse_spin_frames:
        print("[guardrail] mouse spin brake activated")
        state["mouse_spin_until"] = now + args.mouse_spin_brake_seconds
        state["mouse_spin_frames"] = 0
        state["smooth_dx"] = 0.0
        state["mouse_accum_x"] = 0.0

    if now < state.get("mouse_spin_until", 0.0):
        filtered["mouse_dx"] = 0.0
        state["smooth_dx"] = 0.0
        state["mouse_accum_x"] = 0.0

    return filtered



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

    if not isinstance(data, dict):
        return None

    candidates = [
        data,
        data.get("player", {}),
        data.get("position", {}),
    ]

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        if "player_x" in obj and "player_y" in obj:
            return float(obj["player_x"]), float(obj["player_y"])

        if "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"])

    return None


def detect_position_stuck(state, args, final_label, now):
    if not args.guardrails:
        return False

    pos = read_player_position(args.state_json)

    if pos is None:
        return False

    last_pos = state.get("last_position")

    if last_pos is None:
        state["last_position"] = pos
        state["last_position_change_time"] = now
        return False

    dx = pos[0] - last_pos[0]
    dy = pos[1] - last_pos[1]
    dist = (dx * dx + dy * dy) ** 0.5

    if dist >= args.position_min_delta:
        state["last_position"] = pos
        state["last_position_change_time"] = now
        return False

    stuck_time = now - state.get("last_position_change_time", now)

    trying_to_move = (
        "move_forward" in final_label
        or "move_backward" in final_label
        or "strafe_left" in final_label
        or "strafe_right" in final_label
    )

    return trying_to_move and stuck_time >= args.position_stuck_seconds

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

    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--auto-reset-after", type=int, default=2)
    parser.add_argument("--reset-cooldown", type=float, default=10.0)
    parser.add_argument("--reset-key", type=str, default="f9")
    parser.add_argument("--reset-sleep", type=float, default=1.0)

    # Defensive runtime guardrails. These do not make it RL; they only prevent obvious failure loops.
    parser.add_argument("--guardrails", action="store_true")
    parser.add_argument("--screen-stuck-seconds", type=float, default=2.5)
    parser.add_argument("--screen-motion-delta", type=float, default=1.5)

    # Optional real Doom position state. Better than screen-change stuck detection.
    parser.add_argument("--state-json", type=str, default="")
    parser.add_argument("--position-stuck-seconds", type=float, default=2.5)
    parser.add_argument("--position-min-delta", type=float, default=2.0)

    parser.add_argument("--max-same-trace", type=int, default=2)
    parser.add_argument("--trace-repeat-window", type=float, default=12.0)

    # How we bucket nearby frames when deduplicating traces. Increase to
    # group larger windows together; decrease for finer-grained dedupe.
    parser.add_argument("--trace-bucket-size", type=int, default=20)

    # When set, do not block repeated traces even if they exceed
    # --max-same-trace. Useful when you want faithful trace replays and
    # will tolerate longer repeated segments.
    parser.add_argument("--trace-force", action="store_true")

    parser.add_argument("--mouse-spin-threshold", type=float, default=0.75)
    parser.add_argument("--mouse-spin-frames", type=int, default=18)
    parser.add_argument("--mouse-spin-brake-seconds", type=float, default=0.8)

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
        "consecutive_stuck": 0,
        "last_reset_time": 0.0,
        "reset_count": 0,
        "trace_queue": [],
        "last_trace_time": 0.0,
        "trace_id": 0,
        "shoot_streak": 0,

        "last_screen_feature": None,
        "last_screen_change_time": 0.0,
        "last_position": None,
        "last_position_change_time": 0.0,
        "recent_traces": {},
        "probe_queue": [],
        "mouse_spin_frames": 0,
        "last_mouse_sign": 0,
        "mouse_spin_until": 0.0,
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
                    lookup_label = model_action["keyboard_label"]

                    # If the model is low-confidence, do not let nearest-teacher
                    # fallback copy no_op/use forever. Prefer a nearby movement action.
                    if model_action["keyboard_conf"] < args.teacher_conf_threshold:
                        lookup_label = "no_op"

                    teacher_record, teacher_dist = nearest_teacher_for_trace(
                        q_feat,
                        teacher_features,
                        teacher_records,
                        lookup_label,
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

                        if register_trace_or_block(state, args, teacher_record, loop_start):
                            segment = []

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

                probe_action = pop_probe_action(state)

                if probe_action is not None:
                    chosen_action = probe_action
                    source = "probe"
                    reason = f"escape_probe_{len(state.get('probe_queue', []))}"
                    teacher_record = {"keyboard_action": chosen_action["keyboard_label"]}
                    teacher_dist = -2

                # Guardrail: stop low-confidence teacher fallback from spamming use/no_op.
                # This keeps the agent moving instead of standing still tapping use.
                raw_label_for_guard = chosen_action.get("keyboard_label", "no_op")

                if source == "teacher" and reason.startswith("low_conf") and raw_label_for_guard in {"use", "no_op"}:
                    state["teacher_idle_use_streak"] = state.get("teacher_idle_use_streak", 0) + 1
                else:
                    state["teacher_idle_use_streak"] = 0

                if state.get("teacher_idle_use_streak", 0) >= 2:
                    chosen_action = {
                        "keyboard_label": "move_backward",
                        "keyboard_conf": 1.0,
                        "entropy": 0.0,
                        "mouse_dx": 0.0,
                        "mouse_dy": 0.0,
                        "shoot_prob": 0.0,
                    }
                    source = "guard"
                    reason = "blocked_teacher_use_noop"
                    teacher_record = {"keyboard_action": "move_backward"}
                    teacher_dist = -4
                    state["teacher_idle_use_streak"] = 0

                filtered = base.apply_runtime_filters(
                    chosen_action,
                    args,
                    state,
                    loop_start,
                )

                stuck_reason = None

                if detect_screen_stuck(state, args, q_feat, filtered["keyboard_label"], loop_start):
                    stuck_reason = "screen stuck"

                if detect_position_stuck(state, args, filtered["keyboard_label"], loop_start):
                    stuck_reason = "position stuck" if stuck_reason is None else f"{stuck_reason}/position stuck"

                if stuck_reason is not None:
                    state["consecutive_stuck"] = state.get("consecutive_stuck", 0) + 1

                    if args.auto_reset and state["consecutive_stuck"] >= args.auto_reset_after:
                        if maybe_auto_reset(
                            state,
                            args,
                            keyboard,
                            mouse,
                            pressed_keys,
                            loop_start,
                        ):
                            continue

                    print(f"[guardrail] {stuck_reason} detected; using escape probe")
                    make_escape_probe(state)
                    probe_action = pop_probe_action(state)

                    if probe_action is not None:
                        chosen_action = probe_action
                        source = "probe"
                        reason = f"{stuck_reason.replace(' ', '_')}_probe"
                        teacher_record = {"keyboard_action": chosen_action["keyboard_label"]}
                        teacher_dist = -2

                        filtered = base.apply_runtime_filters(
                            chosen_action,
                            args,
                            state,
                            loop_start,
                        )
                else:
                    state["consecutive_stuck"] = 0

                filtered = apply_mouse_spin_guard(state, args, filtered, loop_start)

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