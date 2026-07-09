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

ACTION_TO_BUTTON = {
    "move_forward": "MOVE_FORWARD",
    "move_backward": "MOVE_BACKWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    "strafe_left": "MOVE_LEFT",
    "strafe_right": "MOVE_RIGHT",
    "shoot": "ATTACK",
    "use": "USE",
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


def buttons_to_action(buttons):
    names = []
    for i, v in enumerate(buttons[:8]):
        if int(v):
            names.append(ACTION_NAMES[i])
    return "+".join(names) if names else "no_op"


def actions_to_buttons(active):
    active_buttons = set(ACTION_TO_BUTTON[a] for a in active if a in ACTION_TO_BUTTON)
    return [1 if name in active_buttons else 0 for name in BUTTON_ORDER]


def buttons_to_active(buttons):
    active = []
    for i, v in enumerate(buttons[:8]):
        if int(v):
            active.append(ACTION_NAMES[i])
    return active


def resolve_conflicts(probs, active):
    active = set(active)
    prob_map = {ACTION_NAMES[i]: float(probs[i]) for i in range(min(len(probs), len(ACTION_NAMES)))}

    for a, b in [
        ("turn_left", "turn_right"),
        ("strafe_left", "strafe_right"),
        ("move_forward", "move_backward"),
    ]:
        if a in active and b in active:
            if prob_map.get(a, 0.0) >= prob_map.get(b, 0.0):
                active.remove(b)
            else:
                active.remove(a)

    return sorted(active)


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
            [
                p.parent
                for p in root.glob("session_20*/actions.csv")
                if "clean_wall" not in str(p.parent)
            ],
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
            if "clean_wall" in session.name:
                continue
            try:
                rows = sum(1 for _ in csv_path.open()) - 1
            except Exception:
                rows = -1
            if rows > best_rows:
                best_rows = rows
                best = session
        if best is None:
            raise SystemExit("No raw session found.")
        print(f"[route_agent] longest_raw rows={best_rows}")
        return best

    p = Path(name)
    if p.is_dir():
        return p

    p2 = root / name
    if p2.is_dir():
        return p2

    raise SystemExit(f"Could not find route session: {name}")


def load_route(session):
    csv_path = session / "actions.csv"
    rows = list(csv.DictReader(csv_path.open()))
    route = []

    for row in rows:
        try:
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
        except Exception:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        try:
            angle = float(row.get("angle", "nan"))
        except Exception:
            angle = float("nan")

        buttons = parse_buttons(row)
        action = buttons_to_action(buttons)

        route.append({
            "x": x,
            "y": y,
            "angle": angle,
            "buttons": buttons,
            "action": action,
        })

    if len(route) < 50:
        raise SystemExit(f"Route too short or missing x/y data: {session}")

    return route


def dist2(a, b, x, y):
    dx = a - x
    dy = b - y
    return dx * dx + dy * dy


def nearest_route_index(route, x, y, current_idx, back=30, forward=180):
    lo = max(0, current_idx - back)
    hi = min(len(route), current_idx + forward)

    best_i = current_idx
    best_d = float("inf")

    for i in range(lo, hi):
        d = dist2(route[i]["x"], route[i]["y"], x, y)
        if d < best_d:
            best_d = d
            best_i = i

    return best_i, math.sqrt(best_d)


def next_non_noop_route_action(route, idx, lookahead=40):
    for j in range(idx, min(len(route), idx + lookahead)):
        if route[j]["action"] != "no_op":
            return route[j]["buttons"], route[j]["action"], j
    return route[idx]["buttons"], route[idx]["action"], idx


def upcoming_use(route, idx, lookahead=30):
    for j in range(idx, min(len(route), idx + lookahead)):
        if route[j]["buttons"][7]:
            return True, j
    return False, -1


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
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
    ap.add_argument("--route-session", default="latest", help="latest, longest_raw, folder name, or path")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--fallback-min", type=float, default=0.10)
    ap.add_argument("--route-radius", type=float, default=140.0)
    ap.add_argument("--stuck-steps", type=int, default=18)
    ap.add_argument("--stuck-min-dist", type=float, default=5.0)
    ap.add_argument("--noop-grace", type=int, default=15)
    ap.add_argument("--episode-timeout", type=int, default=5000)
    ap.add_argument("--visible", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    seq_len = int(ckpt.get("seq_len", 64))
    image_size = int(ckpt.get("image_size", 128))

    model = VideoBCNet(num_actions=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    route_session = find_route_session(args.route_session)
    route = load_route(route_session)

    print("[route_agent] checkpoint:", args.checkpoint)
    print("[route_agent] route_session:", route_session)
    print("[route_agent] route frames:", len(route))
    print("[route_agent] device:", device)
    print("[route_agent] seq_len:", seq_len)
    print("[route_agent] threshold:", args.threshold)
    print("[route_agent] route_radius:", args.route_radius)
    print("[route_agent] stuck_steps:", args.stuck_steps)

    game = make_game(args)
    game.init()
    game.new_episode()

    frame_dt = 1.0 / max(1.0, args.fps)
    history = deque(maxlen=seq_len)
    pos_history = deque(maxlen=max(2, args.stuck_steps))

    state = game.get_state()
    if state is None:
        raise SystemExit("No initial game state.")

    first = screen_to_tensor(state.screen_buffer, image_size)
    for _ in range(seq_len):
        history.append(first)

    route_idx = 0
    no_op_count = 0
    step = 0

    try:
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break

            vars_now = list(state.game_variables)
            x = float(vars_now[0]) if len(vars_now) > 0 else 0.0
            y = float(vars_now[1]) if len(vars_now) > 1 else 0.0

            route_idx, route_dist = nearest_route_index(route, x, y, route_idx)
            pos_history.append((x, y))

            stuck = False
            if len(pos_history) >= args.stuck_steps:
                x0, y0 = pos_history[0]
                moved = math.sqrt(dist2(x0, y0, x, y))
                stuck = moved < args.stuck_min_dist

            history.append(screen_to_tensor(state.screen_buffer, image_size))
            model_x = torch.stack(list(history), dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(model_x)
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

            active = [
                ACTION_NAMES[i]
                for i, p in enumerate(probs)
                if float(p) >= args.threshold
            ]

            source = "model"

            # Fallback if model is low confidence.
            if not active:
                top_i = int(np.argmax(probs))
                if float(probs[top_i]) >= args.fallback_min:
                    active = [ACTION_NAMES[top_i]]
                    source = "model_top"
                else:
                    route_buttons, route_action, used_i = next_non_noop_route_action(route, route_idx)
                    active = buttons_to_active(route_buttons)
                    source = f"route_lowconf:{route_action}@{used_i}"

            active = resolve_conflicts(probs, active)

            if not active:
                no_op_count += 1
            else:
                no_op_count = 0

            # If paused/stuck too long, use route memory instead of freezing.
            if stuck and no_op_count >= args.noop_grace:
                route_buttons, route_action, used_i = next_non_noop_route_action(route, route_idx)
                active = buttons_to_active(route_buttons)
                source = f"route_stuck:{route_action}@{used_i}"
                no_op_count = 0

            # If near a known use/door moment in the route, let route memory handle it.
            has_use, use_i = upcoming_use(route, route_idx)
            if has_use and route_dist <= args.route_radius:
                if route[use_i]["buttons"][7] or probs[7] > 0.05:
                    active = buttons_to_active(route[use_i]["buttons"])
                    source = f"route_use:{route[use_i]['action']}@{use_i}"
                    no_op_count = 0

            buttons = actions_to_buttons(active)
            reward = game.make_action(buttons, 1)

            if step % 25 == 0:
                top = sorted(
                    [(ACTION_NAMES[i], round(float(probs[i]), 3)) for i in range(8)],
                    key=lambda z: z[1],
                    reverse=True,
                )[:5]

                print(
                    f"[route_agent] step={step} src={source} "
                    f"route_idx={route_idx}/{len(route)} dist={route_dist:.1f} "
                    f"stuck={stuck} active={'+'.join(active) if active else 'no_op'} "
                    f"top={top} reward={reward} vars={vars_now}"
                )

            step += 1
            if args.visible:
                time.sleep(frame_dt)

    except KeyboardInterrupt:
        print("[route_agent] stopped by user")

    finally:
        game.close()

    print("[route_agent] done")


if __name__ == "__main__":
    main()
