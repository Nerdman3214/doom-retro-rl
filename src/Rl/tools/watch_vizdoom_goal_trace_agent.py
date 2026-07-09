#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import argparse
import csv
import ast
import math
import time

import numpy as np
from PIL import Image

import torch
import vizdoom as vzd

from training.train_vizdoom_video_behavior_clone import VideoBCNet


BUTTON_ORDER = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
]

ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

ACTION_TO_INDEX = {
    "move_forward": 0,
    "move_backward": 1,
    "turn_left": 2,
    "turn_right": 3,
    "strafe_left": 4,
    "strafe_right": 5,
    "shoot": 6,
    "use": 7,
}


def parse_buttons(row):
    raw = str(row.get("buttons", "")).strip()

    if raw:
        try:
            if raw.startswith("["):
                vals = ast.literal_eval(raw)
                vals = [int(float(x)) for x in vals]
                if len(vals) >= 8:
                    return vals[:8]
        except Exception:
            pass

        cleaned = raw.replace(",", " ").replace("[", " ").replace("]", " ")
        parts = [p for p in cleaned.split() if p.strip()]
        if len(parts) >= 8:
            try:
                return [int(float(x)) for x in parts[:8]]
            except Exception:
                pass

    buttons = [0] * 8
    action = str(row.get("action", "no_op")).strip()
    if action and action != "no_op":
        for part in action.split("+"):
            part = part.strip()
            if part in ACTION_TO_INDEX:
                buttons[ACTION_TO_INDEX[part]] = 1
    return buttons


def button_vec(*names):
    buttons = [0] * 8
    for name in names:
        if name in ACTION_TO_INDEX:
            buttons[ACTION_TO_INDEX[name]] = 1
    return buttons


def action_name(buttons):
    names = [ACTION_NAMES[i] for i, v in enumerate(buttons[:8]) if int(v)]
    return "+".join(names) if names else "no_op"


def normalize_angle(a):
    return a % 360.0


def signed_angle_error(target, current):
    return ((target - current + 180.0) % 360.0) - 180.0


def dist(a, b, x, y):
    return math.hypot(a - x, b - y)


def screen_to_tensor(screen_buffer, image_size):
    arr = np.asarray(screen_buffer)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)


def find_route_session(name):
    root = Path("vision_dataset/raw_human_play")

    if name == "latest":
        sessions = sorted(
            [p.parent for p in root.glob("session_20*/actions.csv")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            raise SystemExit("No latest raw session found.")
        return sessions[0]

    if name == "longest_raw":
        best = None
        best_rows = -1
        for csv_path in root.glob("session_20*/actions.csv"):
            session = csv_path.parent
            try:
                rows = sum(1 for _ in csv_path.open()) - 1
            except Exception:
                rows = -1
            if rows > best_rows:
                best_rows = rows
                best = session
        if best is None:
            raise SystemExit("No raw session found.")
        print(f"[goal_trace] longest_raw rows={best_rows}")
        return best

    p = Path(name)
    if p.is_dir():
        return p

    p2 = root / name
    if p2.is_dir():
        return p2

    raise SystemExit(f"Could not find route session: {name}")


def load_route(session, spacing):
    rows = list(csv.DictReader((session / "actions.csv").open()))
    raw = []

    for i, row in enumerate(rows):
        try:
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            angle = float(row.get("angle", "nan"))
        except Exception:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        buttons = parse_buttons(row)

        raw.append({
            "i": i,
            "x": x,
            "y": y,
            "angle": angle if not math.isnan(angle) else 0.0,
            "buttons": buttons,
            "action": action_name(buttons),
            "use": bool(buttons[7]),
        })

    if len(raw) < 50:
        raise SystemExit("Route is too short or missing x/y data.")

    # Build spaced waypoints.
    waypoints = []
    last_x = None
    last_y = None

    for p in raw:
        if last_x is None:
            waypoints.append(p)
            last_x, last_y = p["x"], p["y"]
            continue

        if dist(last_x, last_y, p["x"], p["y"]) >= spacing or p["use"]:
            waypoints.append(p)
            last_x, last_y = p["x"], p["y"]

    if waypoints[-1]["i"] != raw[-1]["i"]:
        waypoints.append(raw[-1])

    return raw, waypoints


def choose_model_buttons(model, history, device, threshold, fallback_min):
    x = torch.stack(list(history), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    buttons = [0] * 8
    for i, p in enumerate(probs[:8]):
        if float(p) >= threshold:
            buttons[i] = 1

    # Resolve opposite controls.
    for a, b in [(2, 3), (4, 5), (0, 1)]:
        if buttons[a] and buttons[b]:
            if probs[a] >= probs[b]:
                buttons[b] = 0
            else:
                buttons[a] = 0

    if not any(buttons):
        top_i = int(np.argmax(probs))
        if float(probs[top_i]) >= fallback_min:
            buttons[top_i] = 1

    return buttons, probs


def steering_buttons(x, y, angle, target, step, args):
    dx = target["x"] - x
    dy = target["y"] - y

    desired = normalize_angle(math.degrees(math.atan2(dy, dx)))
    err = signed_angle_error(desired, angle)

    if args.invert_turn:
        err = -err

    abs_err = abs(err)
    buttons = [0] * 8

    if abs_err > args.hard_turn_deg:
        if err > 0:
            buttons[2] = 1
        else:
            buttons[3] = 1
    elif abs_err > args.soft_turn_deg:
        buttons[0] = 1
        if err > 0:
            buttons[2] = 1
        else:
            buttons[3] = 1
    else:
        buttons[0] = 1

    return buttons, desired, err


def recovery_buttons(step, x=None, y=None, angle=None, target=None, args=None):
    # Goal-directed recovery:
    # 1. Back up briefly to get off the wall.
    # 2. Turn/move toward a future waypoint instead of spinning in place.
    if target is None or args is None or x is None or y is None or angle is None:
        phase = (step // 12) % 4
        if phase == 0:
            return button_vec("move_backward", "turn_left")
        if phase == 1:
            return button_vec("move_backward", "turn_right")
        if phase == 2:
            return button_vec("strafe_left", "turn_left")
        return button_vec("strafe_right", "turn_right")

    cycle = max(1, args.recovery_backup_steps + 24)
    phase = step % cycle

    if phase < args.recovery_backup_steps:
        # Back up and add a small turn toward the future route direction.
        dx = target["x"] - x
        dy = target["y"] - y
        desired = normalize_angle(math.degrees(math.atan2(dy, dx)))
        err = signed_angle_error(desired, angle)
        if args.invert_turn:
            err = -err

        if err > 0:
            return button_vec("move_backward", "turn_left")
        return button_vec("move_backward", "turn_right")

    buttons, _, _ = steering_buttons(x, y, angle, target, step, args)
    return buttons



def enemy_targeted(state, x, y, angle, args):
    # Best-effort enemy/crosshair sense using ViZDoom labels/objects.
    # If labels are unavailable, return False and rely on high shoot confidence only.
    labels = getattr(state, "labels", None)
    if not labels:
        return False

    enemy_words = (
        "zombieman", "shotgun", "imp", "demon", "spectre", "cacodemon",
        "baron", "hell", "lost", "enemy", "monster", "former", "trooper",
    )

    for obj in labels:
        name = str(getattr(obj, "object_name", "")).lower()
        if not any(w in name for w in enemy_words):
            continue

        ex = getattr(obj, "object_position_x", None)
        ey = getattr(obj, "object_position_y", None)

        if ex is None or ey is None:
            # Fallback: if labels exist but no world position, trust that an enemy label is visible.
            return True

        dx = float(ex) - x
        dy = float(ey) - y
        d = math.hypot(dx, dy)
        if d > args.target_distance:
            continue

        desired = normalize_angle(math.degrees(math.atan2(dy, dx)))
        err = abs(signed_angle_error(desired, angle))

        if err <= args.target_angle_deg:
            return True

    return False


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Human-like senses: object/label info when ViZDoom supports it.
    if hasattr(game, "set_labels_buffer_enabled"):
        game.set_labels_buffer_enabled(True)
    if hasattr(game, "set_objects_info_enabled"):
        game.set_objects_info_enabled(True)

    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    for var in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "POSITION_Z"]:
        if hasattr(vzd.GameVariable, var):
            game.add_available_game_variable(getattr(vzd.GameVariable, var))

    return game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vizdoom_video_bc_baseline_seq64_minrows120.pt")
    ap.add_argument("--route-session", default="latest")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--fallback-min", type=float, default=0.10)
    ap.add_argument("--waypoint-spacing", type=float, default=80.0)
    ap.add_argument("--advance-dist", type=float, default=70.0)
    ap.add_argument("--offroute-dist", type=float, default=160.0)
    ap.add_argument("--soft-turn-deg", type=float, default=12.0)
    ap.add_argument("--hard-turn-deg", type=float, default=40.0)
    ap.add_argument("--stuck-steps", type=int, default=18)
    ap.add_argument("--stuck-min-dist", type=float, default=5.0)
    ap.add_argument("--use-radius", type=float, default=90.0)
    ap.add_argument("--episode-timeout", type=int, default=5000)
    ap.add_argument("--visible", action="store_true", default=True)
    ap.add_argument("--goal-lookahead", type=int, default=4)
    ap.add_argument("--recovery-backup-steps", type=int, default=10)
    ap.add_argument("--target-angle-deg", type=float, default=14.0)
    ap.add_argument("--target-distance", type=float, default=900.0)
    ap.add_argument("--allow-untargeted-shoot", action="store_true")
    ap.add_argument("--invert-turn", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    seq_len = int(ckpt.get("seq_len", 64))
    image_size = int(ckpt.get("image_size", 128))

    model = VideoBCNet(num_actions=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    route_session = find_route_session(args.route_session)
    raw_route, waypoints = load_route(route_session, args.waypoint_spacing)

    print("[goal_trace] checkpoint:", args.checkpoint)
    print("[goal_trace] route_session:", route_session)
    print("[goal_trace] raw route frames:", len(raw_route))
    print("[goal_trace] waypoints:", len(waypoints))
    print("[goal_trace] device:", device)
    print("[goal_trace] seq_len:", seq_len)

    game = make_game(args)
    game.init()
    game.new_episode()

    history = deque(maxlen=seq_len)
    pos_history = deque(maxlen=max(2, args.stuck_steps))

    state = game.get_state()
    if state is None:
        raise SystemExit("No initial game state.")

    first = screen_to_tensor(state.screen_buffer, image_size)
    for _ in range(seq_len):
        history.append(first)

    wp_idx = 0
    step = 0
    frame_dt = 1.0 / max(1.0, args.fps)

    try:
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break

            vars_now = list(state.game_variables)
            x = float(vars_now[0]) if len(vars_now) > 0 else 0.0
            y = float(vars_now[1]) if len(vars_now) > 1 else 0.0
            angle = float(vars_now[2]) if len(vars_now) > 2 else 0.0

            pos_history.append((x, y))
            stuck = False
            if len(pos_history) >= args.stuck_steps:
                x0, y0 = pos_history[0]
                moved = dist(x0, y0, x, y)
                stuck = moved < args.stuck_min_dist

            # Advance waypoints only forward.
            while wp_idx < len(waypoints) - 1:
                d = dist(waypoints[wp_idx]["x"], waypoints[wp_idx]["y"], x, y)
                if d <= args.advance_dist:
                    wp_idx += 1
                else:
                    break

            target = waypoints[min(wp_idx, len(waypoints) - 1)]
            target_dist = dist(target["x"], target["y"], x, y)

            # Look ahead so the corridor/door area is not treated like the final goal.
            steer_target = waypoints[min(wp_idx + args.goal_lookahead, len(waypoints) - 1)]

            history.append(screen_to_tensor(state.screen_buffer, image_size))
            model_buttons, probs = choose_model_buttons(
                model,
                history,
                device,
                args.threshold,
                args.fallback_min,
            )

            source = "model"
            buttons = model_buttons

            # Goal tracer takes over when far from the waypoint.
            if target_dist > args.offroute_dist:
                buttons, desired, err = steering_buttons(x, y, angle, steer_target, step, args)
                source = f"goal_steer_ahead:err={err:.1f}"

            # Recovery takes over when not moving.
            if stuck:
                buttons = recovery_buttons(step, x, y, angle, steer_target, args)
                source = "goal_recovery_ahead"

            # If this waypoint is a known use moment, press use near it.
            if target.get("use") and target_dist <= args.use_radius:
                buttons = button_vec("use")
                source = "goal_use"

            # Human-like combat sense: do not shoot randomly unless target is visible
            # or the model is extremely confident.
            if buttons[6] and not args.allow_untargeted_shoot:
                target_ok = enemy_targeted(state, x, y, angle, args)
                shoot_conf = float(probs[6]) if len(probs) > 6 else 0.0

                if not target_ok and shoot_conf < 0.90:
                    buttons[6] = 0
                    if source == "model":
                        source = "shoot_guard"

            reward = game.make_action(buttons, 1)

            if step % 25 == 0:
                top = sorted(
                    [(ACTION_NAMES[i], round(float(probs[i]), 3)) for i in range(8)],
                    key=lambda z: z[1],
                    reverse=True,
                )[:5]

                print(
                    f"[goal_trace] step={step} src={source} "
                    f"wp={wp_idx}/{len(waypoints)} target_dist={target_dist:.1f} "
                    f"stuck={stuck} active={action_name(buttons)} "
                    f"target=({target['x']:.1f},{target['y']:.1f}) "
                    f"pos=({x:.1f},{y:.1f}) angle={angle:.1f} "
                    f"top={top} reward={reward}"
                )

            step += 1
            if args.visible:
                time.sleep(frame_dt)

    except KeyboardInterrupt:
        print("[goal_trace] stopped by user")

    finally:
        game.close()

    print("[goal_trace] done")


if __name__ == "__main__":
    main()
