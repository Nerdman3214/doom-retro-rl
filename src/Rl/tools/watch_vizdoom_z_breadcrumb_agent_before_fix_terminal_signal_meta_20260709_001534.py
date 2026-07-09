#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import argparse
import random
import json
import os
import shutil
import time
import ast
import csv
import math
import time

import vizdoom as vzd


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

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}


def parse_buttons(row):
    raw = str(row.get("buttons", "")).strip()

    if raw:
        try:
            vals = ast.literal_eval(raw)
            vals = [int(float(x)) for x in vals]
            if len(vals) >= 8:
                return vals[:8]
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


def action_name(buttons):
    names = [ACTION_NAMES[i] for i, v in enumerate(buttons[:8]) if int(v)]
    return "+".join(names) if names else "no_op"


def button_vec(*names):
    buttons = [0] * 8
    for name in names:
        if name in ACTION_TO_INDEX:
            buttons[ACTION_TO_INDEX[name]] = 1
    return buttons


def normalize_angle(a):
    return a % 360.0


def signed_angle_error(target, current):
    return ((target - current + 180.0) % 360.0) - 180.0


def dist2d(a, b, x, y):
    return math.hypot(a - x, b - y)


def dist3d(p, x, y, z, z_weight):
    dx = p["x"] - x
    dy = p["y"] - y
    dz = (p["z"] - z) * z_weight
    return math.sqrt(dx * dx + dy * dy + dz * dz)



def has_use_between(waypoints, start, end):
    lo = max(0, min(start, end))
    hi = min(len(waypoints), max(start, end) + 1)

    for k in range(lo, hi):
        if waypoints[k].get("use", False):
            return True

    return False


def route_session_from_arg(name):
    root = Path("vision_dataset/raw_human_play")

    if name == "latest_z":
        sessions = sorted(
            [p.parent for p in root.glob("session_z_*/actions.csv")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            raise SystemExit("No session_z_* route found.")
        return sessions[0]

    if name == "longest_z":
        best = None
        best_rows = -1
        for csv_path in root.glob("session_z_*/actions.csv"):
            try:
                rows = sum(1 for _ in csv_path.open()) - 1
            except Exception:
                rows = -1
            if rows > best_rows:
                best_rows = rows
                best = csv_path.parent
        if best is None:
            raise SystemExit("No session_z_* route found.")
        print(f"[z_breadcrumb] longest_z rows={best_rows}")
        return best

    p = Path(name)
    if p.is_dir():
        return p

    p2 = root / name
    if p2.is_dir():
        return p2

    raise SystemExit(f"Could not find route session: {name}")


def load_route(session, frame_stride):
    rows = list(csv.DictReader((session / "actions.csv").open()))

    raw = []
    for i, row in enumerate(rows):
        if "z" not in row:
            raise SystemExit("This route has no z column. Use session_z_* recorded with record_vizdoom_raw_human_play_z.py")

        try:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            angle = float(row.get("angle", 0.0))
        except Exception:
            continue

        buttons = parse_buttons(row)

        raw.append({
            "raw_i": i,
            "x": x,
            "y": y,
            "z": z,
            "angle": angle,
            "buttons": buttons,
            "action": action_name(buttons),
            "use": bool(buttons[7]),
        })

    if len(raw) < 50:
        raise SystemExit("Route is too short.")

    waypoints = []
    last_z = raw[0]["z"]

    for i, p in enumerate(raw):
        z_changed = abs(p["z"] - last_z) >= 8.0
        should_keep = (
            i == 0
            or i == len(raw) - 1
            or i % frame_stride == 0
            or p["use"]
            or z_changed
        )

        if should_keep:
            waypoints.append(p)
            last_z = p["z"]

    return raw, waypoints



def bc_screen_to_tensor(screen, image_size):
    """Convert VizDoom screen buffer to a 3xHxW float tensor in 0..1."""
    import numpy as np
    import torch
    from PIL import Image

    arr = np.asarray(screen)

    # VizDoom is often C,H,W. Convert to H,W,C.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)

    # Grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Drop alpha or repeat single channel.
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.uint8)

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    img = Image.fromarray(arr).convert("RGB").resize((image_size, image_size), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def bc_override_is_allowed(wp_idx, last_override_step, cooldown_steps=6):
    """Return True when BC is allowed to override again after a cooldown."""
    if cooldown_steps <= 0:
        return True
    if last_override_step is None:
        return True
    return wp_idx >= last_override_step + cooldown_steps


def bc_names_to_buttons(action_names, probs, threshold):
    """Convert BC probabilities to a conservative route-movement action.

    Early BC is not allowed to press use/shoot/back/strafe. This prevents
    hard flicks and bad final-door behavior while we are still testing.
    """
    raw_names = []
    for name, prob in zip(action_names, probs):
        if float(prob) >= threshold:
            raw_names.append(name)

    forbidden = {
        "use",
        "shoot",
        "move_backward",
        "strafe_left",
        "strafe_right",
    }

    if any(name in forbidden for name in raw_names):
        return [], button_vec()

    if "turn_left" in raw_names and "turn_right" in raw_names:
        return [], button_vec()

    if "move_forward" not in raw_names:
        return [], button_vec()

    names = [
        name for name in raw_names
        if name in ("move_forward", "turn_left", "turn_right")
    ]

    return names, button_vec(*names)

def bc_source_is_safe(source):
    """Only allow BC on plain route-following.

    The BC student is still learning movement. Keep all door/use/final/
    terminal/combat/recovery logic under the scripted teacher.
    """
    src = str(source).lower()

    allowed_prefixes = (
        "ghost_route@",
        "route_action",
        "z_steer",
        "noop_to_steer",
    )

    if not src.startswith(allowed_prefixes):
        return False

    blocked = [
        "door",
        "goal",
        "final",
        "terminal",
        "combat",
        "damage",
        "stuck",
        "recover",
        "recovery",
        "use",
        "shoot",
        "magnet",
        "rescue",
        "wall",
        "after_use",
        "cross",
    ]

    return not any(tok in src for tok in blocked)

def steering_buttons(x, y, angle, target, invert_turn=False):
    dx = target["x"] - x
    dy = target["y"] - y

    desired = normalize_angle(math.degrees(math.atan2(dy, dx)))
    err = signed_angle_error(desired, angle)

    if invert_turn:
        err = -err

    buttons = [0] * 8
    abs_err = abs(err)

    if abs_err > 45:
        if err > 0:
            buttons[2] = 1
        else:
            buttons[3] = 1
    elif abs_err > 12:
        buttons[0] = 1
        if err > 0:
            buttons[2] = 1
        else:
            buttons[3] = 1
    else:
        buttons[0] = 1

    return buttons, err


def recovery_buttons(step, x, y, angle, target, invert_turn=False):
    # No random spinning. Back up, then steer toward the next breadcrumb.
    phase = step % 32

    if phase < 10:
        steer, err = steering_buttons(x, y, angle, target, invert_turn)
        if steer[2]:
            return button_vec("move_backward", "turn_left")
        if steer[3]:
            return button_vec("move_backward", "turn_right")
        return button_vec("move_backward")

    steer, err = steering_buttons(x, y, angle, target, invert_turn)
    return steer


def non_noop_route_action(waypoints, idx, lookahead=8):
    for j in range(idx, min(len(waypoints), idx + lookahead)):
        if waypoints[j]["action"] != "no_op":
            return waypoints[j]["buttons"], j
    return waypoints[idx]["buttons"], idx



def find_upcoming_use(waypoints, wp_idx, x, y, z, args):
    hi = min(len(waypoints), wp_idx + args.door_lookahead)

    best = None
    best_d2 = float("inf")

    for j in range(wp_idx, hi):
        p = waypoints[j]

        if not p.get("use", False):
            continue

        if abs(p["z"] - z) > args.z_tolerance:
            continue

        d2 = dist2d(p["x"], p["y"], x, y)

        if d2 < best_d2:
            best = j
            best_d2 = d2

    return best, best_d2


def has_use_button(buttons):
    return len(buttons) > 7 and int(buttons[7]) == 1



def distance_to_route_corridor(ex, ey, ez, waypoints, wp_idx, args):
    best = float("inf")
    hi = min(len(waypoints), wp_idx + args.combat_route_lookahead)

    for j in range(wp_idx, hi):
        p = waypoints[j]

        if abs(p["z"] - ez) > args.z_tolerance:
            continue

        d = math.hypot(float(p["x"]) - ex, float(p["y"]) - ey)

        if d < best:
            best = d

    return best


def find_enemy_target(state, x, y, z, angle, args, waypoints=None, wp_idx=0):
    labels = getattr(state, "labels", None)
    if not labels:
        return None

    def norm_name(name):
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    # Exact/normalized enemy names. Do NOT use loose substring matching like "imp"
    # because "stimpack" contains "imp" and caused ammo waste.
    live_enemy_names = {
        "zombieman",
        "shotgunguy",
        "shotgunner",
        "formerhuman",
        "formerhumansergeant",
        "doomimp",
        "imp",
        "demon",
        "spectre",
        "lostsoul",
        "cacodemon",
        "hellknight",
        "baronofhell",
        "arachnotron",
        "revenant",
        "mancubus",
        "painlemental",
        "archvile",
        "cyberdemon",
        "spidermastermind",
    }

    # Things ViZDoom may label that should never be combat targets.
    non_enemy_names = {
        "stimpack",
        "medikit",
        "healthbonus",
        "armorbonus",
        "greenarmor",
        "bluearmor",
        "securityarmor",
        "combatarmor",
        "clip",
        "clipbox",
        "shell",
        "shellbox",
        "rocket",
        "rocketbox",
        "cell",
        "cellpack",
        "backpack",
        "shotgun",
        "chaingun",
        "chainsaw",
        "supershotgun",
        "plasmagun",
        "rocketlauncher",
        "bfg9000",
        "bluekeycard",
        "redkeycard",
        "yellowkeycard",
        "bluekey",
        "redkey",
        "yellowkey",
        "barrel",
        "explosivebarrel",
        "teleportfog",
        "blood",
        "gib",
        "corpse",
    }

    dead_words = (
        "dead",
        "corpse",
        "gib",
        "blood",
        "death",
        "dying",
        "remains",
        "body",
    )

    best = None

    for obj in labels:
        raw_name = str(getattr(obj, "object_name", "")).lower()
        name = norm_name(raw_name)

        if not name:
            continue

        # Reject dead bodies/death animation/corpse labels.
        if any(w in name for w in dead_words):
            continue

        # Reject known pickups/items.
        if name in non_enemy_names:
            continue

        # Accept only known live enemy names.
        # This prevents "stimpack" from matching "imp".
        if name not in live_enemy_names:
            continue

        ex = getattr(obj, "object_position_x", None)
        ey = getattr(obj, "object_position_y", None)
        ez = getattr(obj, "object_position_z", z)

        if ex is None or ey is None:
            continue

        dx = float(ex) - x
        dy = float(ey) - y
        dz = float(ez) - z

        d2 = math.hypot(dx, dy)

        if d2 > args.combat_distance:
            continue

        # Route-only combat:
        # only fight enemies that are on or near the planned route corridor.
        # This prevents the agent from chasing side-room enemies or getting distracted.
        if waypoints is not None:
            route_d = distance_to_route_corridor(float(ex), float(ey), float(ez), waypoints, wp_idx, args)
            if route_d > args.combat_route_corridor:
                continue

        # Ignore enemies on very different vertical levels unless they are close.
        if abs(dz) > 96 and d2 > 256:
            continue

        desired = normalize_angle(math.degrees(math.atan2(dy, dx)))
        err = signed_angle_error(desired, angle)

        if args.invert_turn:
            err = -err

        score = abs(err) + (d2 * 0.01)

        target = {
            "name": name,
            "raw_name": raw_name,
            "x": float(ex),
            "y": float(ey),
            "z": float(ez),
            "d2": d2,
            "dz": dz,
            "err": err,
            "score": score,
        }

        if best is None or target["score"] < best["score"]:
            best = target

    return best


def route_correct_to_future_waypoint(waypoints, wp_idx, x, y, z, args, min_advance=None):
    """Return a safer future waypoint index on the same floor.

    This is not a new helper action. It only corrects the route target when
    the current local waypoint is clearly blocked by a wall/door edge.
    """
    if not waypoints:
        return wp_idx, float("inf")

    if min_advance is None:
        min_advance = args.route_correct_min_advance

    start = min(len(waypoints) - 1, max(wp_idx + min_advance, wp_idx + 1))
    hi = min(len(waypoints), wp_idx + args.route_correct_lookahead)

    best_idx = wp_idx
    best_d2 = float("inf")

    for j in range(start, hi):
        pt = waypoints[j]

        if abs(pt["z"] - z) > args.z_tolerance:
            continue

        cand_d2 = dist2d(pt["x"], pt["y"], x, y)

        if cand_d2 < best_d2:
            best_d2 = cand_d2
            best_idx = j

    return best_idx, best_d2


def route_resync_nearby_waypoint(waypoints, wp_idx, x, y, z, args):
    """Find a nearby waypoint around the current route index.

    Unlike the earlier future-only correction, this can move slightly backward
    or forward. That helps when the agent misses a door edge and needs to
    reacquire the real route instead of jumping across a wall.
    """
    if not waypoints:
        return wp_idx, float("inf")

    lo = max(0, wp_idx - args.safe_loop_search_back)
    hi = min(len(waypoints), wp_idx + args.safe_loop_search_ahead)

    best_idx = wp_idx
    best_d2 = float("inf")

    for j in range(lo, hi):
        pt = waypoints[j]

        if abs(pt["z"] - z) > args.z_tolerance:
            continue

        cand_d2 = dist2d(pt["x"], pt["y"], x, y)

        if cand_d2 < best_d2:
            best_d2 = cand_d2
            best_idx = j

    return best_idx, best_d2


def waypoint_has_use(wp):
    action = str(wp.get("action", "")).lower()
    buttons = str(wp.get("buttons", "")).lower()
    active = str(wp.get("active", "")).lower()

    if "use" in action or "use" in buttons or "use" in active:
        return True

    raw = wp.get("button_vec", wp.get("buttons_vec", None))
    if isinstance(raw, (list, tuple)) and len(raw) > 7:
        return bool(raw[7])

    return False


def ghost_anchor_index(waypoints, wp_idx, x, y, z, args):
    """Closest same-floor waypoint ahead of current progress.

    This is the GPS anchor. It never searches behind the current route index,
    so it avoids the old backwards loops like wp 127 -> 110.
    """
    if not waypoints:
        return wp_idx, float("inf")

    lo = max(0, wp_idx)
    hi = min(len(waypoints), wp_idx + args.ghost_search_ahead)

    best_idx = wp_idx
    best_d2 = float("inf")

    for j in range(lo, hi):
        pt = waypoints[j]

        if abs(pt["z"] - z) > args.z_tolerance:
            continue

        d2 = dist2d(pt["x"], pt["y"], x, y)

        if d2 < best_d2:
            best_d2 = d2
            best_idx = j

    return best_idx, best_d2


def waypoint_has_shoot(wp):
    action = str(wp.get("action", "")).lower()
    buttons = str(wp.get("buttons", "")).lower()
    active = str(wp.get("active", "")).lower()

    if "shoot" in action or "attack" in action or "fire" in action:
        return True

    if "shoot" in buttons or "attack" in buttons or "fire" in buttons:
        return True

    if "shoot" in active or "attack" in active or "fire" in active:
        return True

    raw = wp.get("button_vec", wp.get("buttons_vec", None))
    if isinstance(raw, (list, tuple)) and len(raw) > 6:
        return bool(raw[6])

    return False


def ghost_next_shoot_index(waypoints, wp_idx, args):
    hi = min(len(waypoints), wp_idx + args.ghost_shoot_lookahead)

    for j in range(wp_idx, hi):
        if waypoint_has_shoot(waypoints[j]):
            return j

    return None


def recorded_use_angle_error(use_pt, x, y, angle, args):
    use_d2 = dist2d(use_pt["x"], use_pt["y"], x, y)

    if use_d2 <= args.ghost_use_recorded_angle_radius:
        desired_angle = normalize_angle(float(use_pt.get("angle", angle)))
    else:
        desired_angle = normalize_angle(
            math.degrees(math.atan2(use_pt["y"] - y, use_pt["x"] - x))
        )

    err = signed_angle_error(desired_angle, angle)

    if args.invert_turn:
        err = -err

    return err, desired_angle, use_d2


def ghost_last_use_index(waypoints):
    for j in range(len(waypoints) - 1, -1, -1):
        if waypoint_has_use(waypoints[j]):
            return j
    return len(waypoints) - 1


def ghost_nearby_use_index(waypoints, wp_idx, x, y, z, args):
    lo = max(0, wp_idx - args.ghost_stuck_door_backtrack)
    hi = min(len(waypoints), wp_idx + args.ghost_stuck_door_lookahead)

    best_idx = None
    best_d2 = float("inf")

    for j in range(lo, hi):
        pt = waypoints[j]

        if not waypoint_has_use(pt):
            continue

        if abs(pt["z"] - z) > args.z_tolerance:
            continue

        d2 = dist2d(pt["x"], pt["y"], x, y)

        if d2 <= args.ghost_stuck_door_use_radius and d2 < best_d2:
            best_idx = j
            best_d2 = d2

    return best_idx, best_d2


def ghost_next_use_index(waypoints, wp_idx, args):
    hi = min(len(waypoints), wp_idx + args.ghost_use_lookahead)

    for j in range(wp_idx, hi):
        if waypoint_has_use(waypoints[j]):
            return j

    return None


def turn_only_buttons(err):
    buttons = [0] * 8
    if err > 0:
        buttons[2] = 1
    else:
        buttons[3] = 1
    return buttons


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Combat sense: enable labels/object info when ViZDoom supports it.
    if hasattr(game, "set_labels_buffer_enabled"):
        game.set_labels_buffer_enabled(True)
    if hasattr(game, "set_objects_info_enabled"):
        game.set_objects_info_enabled(True)

    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    # Order matters.
    for name in ["POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "HEALTH"]:
        game.add_available_game_variable(getattr(vzd.GameVariable, name))

    return game




def _ai_record_action_from_buttons(buttons):
    names = [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "shoot",
        "use",
    ]

    try:
        active = [
            name for name, pressed in zip(names, buttons)
            if int(pressed) == 1
        ]
    except Exception:
        active = []

    return "+".join(active) if active else "no_op"


def _ai_record_save_frame(game, out_path):
    try:
        import numpy as np
        from PIL import Image

        st = game.get_state()
        if st is None or getattr(st, "screen_buffer", None) is None:
            return ""

        arr = np.asarray(st.screen_buffer)

        # ViZDoom often returns C,H,W.
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)

        Image.fromarray(arr.astype("uint8")).save(out_path)
        return str(out_path)
    except Exception as e:
        print(f"[ai_record] frame save failed: {e}")
        return ""


def _ai_record_write_csv(path, rows):
    import csv

    fieldnames = [
        "frame_path", "frame", "action", "buttons",
        "x", "y", "z", "angle",
        "health", "ammo", "kill_count", "item_count",
        "done", "truncated", "timestamp", "source",
        "wp_idx", "target_x", "target_y", "target_z",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-session", default="longest_z")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--save-ai-success", action="store_true")
    ap.add_argument("--ai-success-dir", default="vision_dataset/success_ai_play")
    ap.add_argument("--ai-success-min-wp-frac", type=float, default=0.98)
    ap.add_argument("--advance-dist", type=float, default=35)
    ap.add_argument("--offroute-dist", type=float, default=90)
    ap.add_argument("--enable-ghost-route", action="store_true", default=True)
    ap.add_argument("--disable-ghost-route", action="store_false", dest="enable_ghost_route")
    ap.add_argument("--ghost-search-ahead", type=int, default=45)
    ap.add_argument("--ghost-lookahead", type=int, default=8)
    ap.add_argument("--ghost-corridor-radius", type=float, default=140)
    ap.add_argument("--ghost-turn-only-deg", type=float, default=22)
    ap.add_argument("--ghost-use-lookahead", type=int, default=25)
    ap.add_argument("--ghost-shoot-lookahead", type=int, default=16)
    ap.add_argument("--ghost-shoot-radius", type=float, default=160)
    ap.add_argument("--ghost-shoot-face-deg", type=float, default=28)
    ap.add_argument("--ghost-shoot-hold-steps", type=int, default=5)
    ap.add_argument("--ghost-shoot-cooldown-steps", type=int, default=28)
    ap.add_argument("--ghost-use-radius", type=float, default=64)
    ap.add_argument("--ghost-stuck-door-use-radius", type=float, default=115)
    ap.add_argument("--ghost-stuck-door-backtrack", type=int, default=25)
    ap.add_argument("--ghost-stuck-door-lookahead", type=int, default=35)
    ap.add_argument("--ghost-stuck-door-cross-steps", type=int, default=45)
    ap.add_argument("--ghost-use-face-deg", type=float, default=24)
    ap.add_argument("--ghost-use-recorded-angle-radius", type=float, default=140)
    ap.add_argument("--ghost-stuck-door-face-deg", type=float, default=18)
    ap.add_argument("--ghost-final-door-min-wp", type=int, default=214)
    ap.add_argument("--ghost-final-force-frac", type=float, default=0.96)
    ap.add_argument("--ghost-final-force-radius", type=float, default=135)
    ap.add_argument("--ghost-final-force-face-deg", type=float, default=10)
    ap.add_argument("--ghost-final-force-cross-steps", type=int, default=35)
    ap.add_argument("--final-wall-x", type=float, default=-368.0)
    ap.add_argument("--final-wall-y", type=float, default=1424.0)
    ap.add_argument("--final-wall-radius", type=float, default=140)
    ap.add_argument("--final-cross-x", type=float, default=-368.0)
    ap.add_argument("--final-cross-y", type=float, default=1308.5)
    ap.add_argument("--final-cross-face-deg", type=float, default=8)
    ap.add_argument("--final-wall-use-steps", type=int, default=8)
    ap.add_argument("--final-wall-cross-steps", type=int, default=80)
    ap.add_argument("--final-wall-rescue-steps", type=int, default=55)
    ap.add_argument("--final-wall-rescue-back-steps", type=int, default=8)
    ap.add_argument("--final-rescue-door-x", type=float, default=-240.0)
    ap.add_argument("--final-rescue-door-y", type=float, default=1416.0)
    ap.add_argument("--final-rescue-door-radius", type=float, default=28.0)
    ap.add_argument("--final-rescue-door-use-steps", type=int, default=12)
    ap.add_argument("--final-rescue-door-cross-steps", type=int, default=45)
    ap.add_argument("--final-rescue-door-face-deg", type=float, default=45.0)
    ap.add_argument("--final-wall-rescue-trigger-y", type=float, default=1410.0)
    ap.add_argument("--ghost-use-hold-steps", type=int, default=6)
    ap.add_argument("--ghost-after-use-cross-steps", type=int, default=45)
    ap.add_argument("--ghost-cross-lookahead", type=int, default=8)
    ap.add_argument("--ghost-use-cooldown-steps", type=int, default=90)
    ap.add_argument("--ghost-final-stuck-use-radius", type=float, default=150)
    ap.add_argument("--ghost-final-min-wp", type=int, default=303)
    ap.add_argument("--terminal-button-x", type=float, default=-368.0)
    ap.add_argument("--terminal-button-y", type=float, default=1288.0)
    ap.add_argument("--terminal-button-z", type=float, default=-128.0)
    ap.add_argument("--terminal-min-wp", type=int, default=316)
    ap.add_argument("--terminal-activation-radius", type=float, default=55)
    ap.add_argument("--terminal-use-radius", type=float, default=70)
    ap.add_argument("--terminal-face-deg", type=float, default=22)
    ap.add_argument("--terminal-magnet-min-wp", type=int, default=254)
    ap.add_argument("--terminal-magnet-radius", type=float, default=180)
    ap.add_argument("--terminal-magnet-max-y", type=float, default=1360.0)
    ap.add_argument("--terminal-magnet-use-radius", type=float, default=18)
    ap.add_argument("--terminal-magnet-face-deg", type=float, default=7)
    ap.add_argument("--terminal-use-lock-steps", type=int, default=18)
    ap.add_argument("--terminal-use-lock-face-deg", type=float, default=24)
    ap.add_argument("--terminal-stuck-recover-steps", type=int, default=28)
    ap.add_argument("--terminal-stuck-back-steps", type=int, default=8)
    ap.add_argument("--terminal-stuck-min-wp-frac", type=float, default=0.97)
    ap.add_argument("--safe-loop-correct-steps", type=int, default=90)
    ap.add_argument("--safe-loop-near-radius", type=float, default=85)
    ap.add_argument("--safe-loop-near-advance", type=int, default=4)
    ap.add_argument("--safe-loop-search-back", type=int, default=18)
    ap.add_argument("--safe-loop-search-ahead", type=int, default=40)
    ap.add_argument("--safe-loop-max-d2", type=float, default=130)
    ap.add_argument("--safe-loop-max-worse", type=float, default=25)
    ap.add_argument("--goal-door-preface-radius", type=float, default=230)
    ap.add_argument("--goal-door-use-radius", type=float, default=165)
    ap.add_argument("--goal-door-face-deg", type=float, default=24)
    ap.add_argument("--z-tolerance", type=float, default=24)
    ap.add_argument("--z-weight", type=float, default=2.5)
    ap.add_argument("--z-resync-radius", type=float, default=260)
    ap.add_argument("--max-z-resync-lookahead", type=int, default=120)
    ap.add_argument("--stuck-steps", type=int, default=18)
    ap.add_argument("--route-correct-stuck-steps", type=int, default=45)
    ap.add_argument("--route-loop-correct-steps", type=int, default=120)
    ap.add_argument("--route-loop-min-improve", type=float, default=8)
    ap.add_argument("--route-loop-correct-radius", type=float, default=260)
    ap.add_argument("--route-loop-correct-advance", type=int, default=14)
    ap.add_argument("--route-correct-lookahead", type=int, default=90)
    ap.add_argument("--route-correct-min-advance", type=int, default=8)
    ap.add_argument("--route-correct-radius", type=float, default=150)
    ap.add_argument("--door-stuck-correct-steps", type=int, default=30)
    ap.add_argument("--door-stuck-correct-advance", type=int, default=10)
    ap.add_argument("--stuck-min-dist", type=float, default=6)
    ap.add_argument("--door-lookahead", type=int, default=90)
    ap.add_argument("--door-radius", type=float, default=180)
    ap.add_argument("--door-use-distance", type=float, default=42)
    ap.add_argument("--door-face-deg", type=float, default=25)
    ap.add_argument("--door-stuck-use-radius", type=float, default=160)
    ap.add_argument("--door-stuck-use-face-deg", type=float, default=45)
    ap.add_argument("--door-wall-recovery-radius", type=float, default=190)
    ap.add_argument("--door-hold-steps", type=int, default=18)
    ap.add_argument("--door-advance-after-use", type=int, default=0)
    ap.add_argument("--door-cross-steps", type=int, default=55)
    ap.add_argument("--door-lock-radius", type=float, default=190)
    ap.add_argument("--door-allow-behind-wp", type=int, default=2)
    ap.add_argument("--forget-passed-door-wps", type=int, default=12)
    ap.add_argument("--post-door-resync-radius", type=float, default=260)
    ap.add_argument("--post-door-resync-lookahead", type=int, default=120)
    ap.add_argument("--door-cross-min-steps", type=int, default=18)
    ap.add_argument("--door-cross-stop-radius", type=float, default=70)
    ap.add_argument("--door-stuck-only", action="store_true")
    ap.add_argument("--goal-lookback", type=int, default=18)
    ap.add_argument("--goal-use-radius", type=float, default=70)
    ap.add_argument("--final-objective-radius", type=float, default=180)
    ap.add_argument("--final-objective-use-radius", type=float, default=155)
    ap.add_argument("--final-objective-face-deg", type=float, default=28)
    ap.add_argument("--final-objective-stuck-radius", type=float, default=220)
    ap.add_argument("--goal-stuck-use-radius", type=float, default=190)
    ap.add_argument("--goal-stuck-face-deg", type=float, default=55)
    ap.add_argument("--goal-wall-recover-radius", type=float, default=230)
    ap.add_argument("--goal-button-lock-radius", type=float, default=160)
    ap.add_argument("--goal-button-use-distance", type=float, default=34)
    ap.add_argument("--goal-button-face-deg", type=float, default=30)
    ap.add_argument("--goal-button-nudge-steps", type=int, default=8)
    ap.add_argument("--goal-face-deg", type=float, default=35)
    ap.add_argument("--goal-use-hold-steps", type=int, default=25)
    ap.add_argument("--goal-wait-after-use", type=int, default=30)
    ap.add_argument("--combat-distance", type=float, default=900)
    ap.add_argument("--combat-shoot-angle", type=float, default=18)
    ap.add_argument("--combat-turn-angle", type=float, default=70)
    ap.add_argument("--combat-shoot-hold", type=int, default=4)
    ap.add_argument("--combat-close-distance", type=float, default=300)
    ap.add_argument("--combat-route-max-d2", type=float, default=260)
    ap.add_argument("--combat-route-corridor", type=float, default=420)
    ap.add_argument("--combat-route-lookahead", type=int, default=120)
    ap.add_argument("--route-combat-distance", type=float, default=900)
    ap.add_argument("--route-combat-shoot-angle", type=float, default=18)
    ap.add_argument("--route-combat-turn-angle", type=float, default=140)
    ap.add_argument("--route-combat-max-steps", type=int, default=160)
    ap.add_argument("--damage-scan-steps", type=int, default=35)
    ap.add_argument("--damage-scan-health-drop", type=float, default=2.0)
    ap.add_argument("--disable-combat", action="store_true")
    ap.add_argument("--route-action-radius", type=float, default=110)
    ap.add_argument("--episode-timeout", type=int, default=8000)
    ap.add_argument("--invert-turn", action="store_true")

    ap.add_argument("--bc-checkpoint", default="", help="Optional behavior cloning checkpoint to mix into safe route movement.")
    ap.add_argument("--bc-mix", type=float, default=0.05, help="Probability of using BC action during safe route movement.")
    ap.add_argument("--bc-threshold", type=float, default=0.5, help="Sigmoid threshold for BC multi-button predictions.")
    ap.add_argument("--bc-min-wp", type=int, default=20, help="Earliest waypoint where BC may override.")
    ap.add_argument("--bc-max-wp", type=int, default=230, help="Latest waypoint where BC may override.")
    ap.add_argument("--bc-device", default="cuda", help="BC device: cuda or cpu.")
    ap.add_argument("--bc-debug-every", type=int, default=50, help="Print BC debug every N BC overrides; 0 disables.")

    args = ap.parse_args()

    bc_model = None
    bc_action_names = []
    bc_seq_len = 8
    bc_image_size = 84
    bc_frame_buffer = None
    bc_override_count = 0
    bc_last_override_step = None

    if args.bc_checkpoint:
        import torch
        from training.train_vizdoom_video_behavior_clone import VideoBCNet

        bc_device = args.bc_device
        if bc_device == "cuda" and not torch.cuda.is_available():
            bc_device = "cpu"

        bc_ckpt = torch.load(args.bc_checkpoint, map_location=bc_device)
        bc_seq_len = int(bc_ckpt.get("seq_len", 8))
        bc_image_size = int(bc_ckpt.get("image_size", 84))
        bc_action_names = list(bc_ckpt["action_names"])

        bc_model = VideoBCNet(num_actions=len(bc_action_names))
        bc_model.load_state_dict(bc_ckpt["model"])
        bc_model.to(bc_device)
        bc_model.eval()

        bc_frame_buffer = deque(maxlen=bc_seq_len)

        print(
            f"[bc] loaded checkpoint={args.bc_checkpoint} "
            f"device={bc_device} seq_len={bc_seq_len} image_size={bc_image_size} "
            f"actions={bc_action_names} mix={args.bc_mix} "
            f"wp_range={args.bc_min_wp}-{args.bc_max_wp}"
        )

    # Safety clamp: combat should not pull the route controller away from the path.
    # This prevents accidental values like --combat-close-distance 3000 from
    # making the agent chase/fight everything instead of finishing doors/routes.
    args.combat_close_distance = min(float(args.combat_close_distance), 500.0)
    args.combat_distance = min(float(args.combat_distance), 900.0)

    route_session = route_session_from_arg(args.route_session)
    raw, waypoints = load_route(route_session, args.frame_stride)

    print("[z_breadcrumb] route_session:", route_session)
    print("[z_breadcrumb] raw frames:", len(raw))
    print("[z_breadcrumb] waypoints:", len(waypoints))
    print("[z_breadcrumb] z range:", min(p["z"] for p in raw), "to", max(p["z"] for p in raw))

    ai_record_dir = None
    ai_record_frames_dir = None
    ai_record_rows = []
    ai_record_max_wp = 0
    ai_record_terminal_signal = False
    ai_record_start_time = time.time()

    if args.save_ai_success:
        root = Path(args.ai_success_dir)
        root.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        ai_record_dir = root / f"session_ai_tmp_{stamp}_{os.getpid()}"
        ai_record_frames_dir = ai_record_dir / "frames"
        ai_record_frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ai_record] recording candidate AI episode: {ai_record_dir}")


    game = make_game(args)
    game.init()
    game.new_episode()

    wp_idx = 0
    step = 0
    use_hold = 0
    shoot_hold = 0
    door_advance_countdown = 0
    door_lock_idx = None
    last_door_lock_idx = None
    door_cross_countdown = 0
    door_cross_min_countdown = 0
    goal_use_hold = 0
    goal_wait_countdown = 0
    goal_nudge_countdown = 0
    goal_done = False
    route_combat_steps = 0
    damage_scan_countdown = 0
    last_health_for_scan = None
    pos_hist = deque(maxlen=args.stuck_steps)
    ghost_shoot_cooldown = 0
    final_wall_use_countdown = 0
    final_wall_cross_countdown = 0
    final_wall_rescue_countdown = 0
    final_rescue_door_use_countdown = 0
    final_rescue_door_cross_countdown = 0
    terminal_stuck_recover_countdown = 0
    terminal_use_lock_countdown = 0
    ai_record_true_episode_finished = False
    ai_record_final_action_used = False
    ai_record_final_action_name = ""
    final_use_nudge_countdown = 0
    final_use_attempt_count = 0
    final_turn_pulse_cooldown = 0
    final_last_use_step = -999999
    last_ghost_shoot_idx = -1
    ghost_use_cooldown = 0
    ghost_cross_countdown = 0
    last_ghost_use_idx = -1
    route_loop_wp = None
    route_loop_count = 0
    route_loop_best_d2 = float("inf")
    route_stuck_count = 0
    route_stuck_wp = None
    door_stuck_count = 0
    sleep_dt = 1.0 / max(1.0, args.fps)

    try:
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break

            vars_now = list(state.game_variables)
            x = float(vars_now[0])
            y = float(vars_now[1])
            z = float(vars_now[2])
            angle = float(vars_now[3])
            health = float(vars_now[4]) if len(vars_now) > 4 else 0.0

            pos_hist.append((x, y, z))

            stuck = False
            if len(pos_hist) >= args.stuck_steps:
                x0, y0, z0 = pos_hist[0]
                moved = math.sqrt((x - x0) ** 2 + (y - y0) ** 2 + ((z - z0) * args.z_weight) ** 2)
                stuck = moved < args.stuck_min_dist

            # Move forward only through route order, but handle elevation transitions.
            # If we already fell/dropped below an intermediate z waypoint, skip it.
            while wp_idx < len(waypoints) - 1:
                p = waypoints[wp_idx]
                d2 = dist2d(p["x"], p["y"], x, y)
                same_floor = abs(p["z"] - z) <= args.z_tolerance

                # Doom elevators/drops can move through intermediate z values quickly.
                # If the current z is already lower than this target z, don't keep chasing it.
                descended_past = z < (p["z"] - args.z_tolerance)

                if (same_floor and d2 <= args.advance_dist) or descended_past:
                    wp_idx += 1
                else:
                    break

            # Z-resync: after an elevator/drop, jump forward to a future waypoint
            # on the current floor if it is spatially nearby. This prevents the
            # agent from circling around an old intermediate-height waypoint.
            best_idx = wp_idx
            best_d2 = float("inf")
            hi = min(len(waypoints), wp_idx + args.max_z_resync_lookahead)

            for j in range(wp_idx, hi):
                # Do not resync past a door/use event.
                # Doors are route events and must be completed before skipping ahead.
                if j > wp_idx and has_use_between(waypoints, wp_idx, j):
                    break

                p = waypoints[j]
                if abs(p["z"] - z) > args.z_tolerance:
                    continue

                d2 = dist2d(p["x"], p["y"], x, y)
                if d2 <= args.z_resync_radius and d2 < best_d2:
                    best_idx = j
                    best_d2 = d2

            if best_idx > wp_idx:
                print(
                    f"[z_breadcrumb] z_resync wp {wp_idx}->{best_idx} "
                    f"d2={best_d2:.1f} z={z:.1f}"
                )
                wp_idx = best_idx

            target = waypoints[min(wp_idx, len(waypoints) - 1)]
            target_d3 = dist3d(target, x, y, z, args.z_weight)
            target_d2 = dist2d(target["x"], target["y"], x, y)
            z_diff = target["z"] - z
            same_floor = abs(z_diff) <= args.z_tolerance

            source = "route_action"
            buttons = target["buttons"]

            # If the next breadcrumb is on a different floor, don't steer directly through walls.
            # Use recorded route actions around the transition.
            if not same_floor:
                buttons, used_i = non_noop_route_action(waypoints, wp_idx)
                source = f"z_transition_route_action@{used_i}"

            # If far from the breadcrumb on the same floor, steer back to it.
            elif target_d2 > args.offroute_dist:
                buttons, err = steering_buttons(x, y, angle, target, args.invert_turn)
                source = f"z_steer:err={err:.1f}"

            # If close to the breadcrumb, mostly replay human route actions instead of shortcut steering.
            elif target_d2 <= args.route_action_radius:
                buttons, used_i = non_noop_route_action(waypoints, wp_idx)
                source = f"route_action@{used_i}"

            # If route action is no_op while not progressing, steer.
            if not any(buttons) and target_d2 > args.advance_dist:
                buttons, err = steering_buttons(x, y, angle, target, args.invert_turn)
                source = f"noop_to_steer:err={err:.1f}"

            if stuck:
                buttons = recovery_buttons(step, x, y, angle, target, args.invert_turn)
                source = "z_recovery"

            # Route replay should only control movement.
            # Shooting must come from live combat labels, not old recorded demo shots.
            # USE must come from door/goal logic, not old route replay.
            if str(source).startswith("route_action"):
                if len(buttons) > 6:
                    buttons[6] = 0  # shoot
                if len(buttons) > 7:
                    buttons[7] = 0  # use

            # Forget doors that are clearly behind the current route position.
            # This prevents 180-degree turns back to old doors after resync.
            if (
                last_door_lock_idx is not None
                and wp_idx > last_door_lock_idx + args.forget_passed_door_wps
            ):
                last_door_lock_idx = None

            if (
                door_lock_idx is not None
                and wp_idx > door_lock_idx + args.forget_passed_door_wps
            ):
                print(
                    f"[z_breadcrumb] forget_old_door lock={door_lock_idx} wp={wp_idx}"
                )
                door_lock_idx = None
                door_cross_countdown = 0
                use_hold = 0

            # Door/use awareness with lock mode:
            # Once a door is detected, keep working that same door until we use it,
            # then move forward through it before returning to normal route logic.
            use_idx, use_d2 = find_upcoming_use(waypoints, wp_idx, x, y, z, args)

            if (
                door_lock_idx is None
                and use_idx is not None
                and use_idx >= max(0, wp_idx - args.door_allow_behind_wp)
                and use_d2 <= args.door_lock_radius
            ):
                door_lock_idx = use_idx
                last_door_lock_idx = use_idx
                source = f"door_lock_start@{door_lock_idx}:d2={use_d2:.1f}"

            elif (
                door_lock_idx is None
                and use_idx is not None
                and use_idx < max(0, wp_idx - args.door_allow_behind_wp)
                and use_d2 <= args.door_lock_radius
            ):
                source = f"ignore_old_door@{use_idx}:wp={wp_idx}:d2={use_d2:.1f}"

            current_or_route_use = has_use_button(buttons)

            if door_advance_countdown > 0:
                door_advance_countdown -= 1
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                source = "door_force_advance"

            should_use = False

            final_target = waypoints[-1]
            final_d2_for_door = dist2d(final_target["x"], final_target["y"], x, y)
            final_same_floor_for_door = abs(final_target["z"] - z) <= args.z_tolerance
            near_final_button_for_door = (
                wp_idx >= len(waypoints) - args.goal_lookback
                and final_same_floor_for_door
                and final_d2_for_door <= max(args.goal_use_radius, 90)
            )

            # Highest-priority final goal/button logic.
            # Treat the final objective like a special door/button:
            # physically approach it, face it, then press USE.
            final_goal = waypoints[-1]
            goal_d2_now = dist2d(final_goal["x"], final_goal["y"], x, y)
            goal_same_floor_now = abs(final_goal["z"] - z) <= args.z_tolerance
            goal_near_end_now = wp_idx >= max(0, len(waypoints) - args.goal_lookback)

            goal_priority_active = (
                not goal_done
                and goal_same_floor_now
                and goal_near_end_now
                and goal_d2_now <= args.goal_button_lock_radius
            )

            if goal_priority_active:
                # Goal overrides door mode so we do not spam door_hold forever.
                door_cross_countdown = 0
                door_cross_min_countdown = 0
                door_lock_idx = None
                door_advance_countdown = 0
                use_hold = 0

                goal_err = signed_angle_error(final_goal["angle"], angle)
                if args.invert_turn:
                    goal_err = -goal_err

                if goal_wait_countdown > 0:
                    goal_wait_countdown -= 1
                    buttons = [0] * 8
                    source = "goal_priority_wait"

                    if goal_wait_countdown == 0:
                        goal_done = True

                elif goal_use_hold > 0:
                    goal_use_hold -= 1
                    buttons = button_vec("use")
                    source = f"goal_priority_use_hold:d2={goal_d2_now:.1f}:err={goal_err:.1f}"

                    if goal_use_hold == 0:
                        goal_wait_countdown = args.goal_wait_after_use

                elif goal_nudge_countdown > 0:
                    # Small forward nudge after facing, useful when the button is
                    # just out of reach or the agent is slightly off the line.
                    goal_nudge_countdown -= 1
                    buttons = button_vec("move_forward")
                    source = f"goal_priority_nudge:d2={goal_d2_now:.1f}:err={goal_err:.1f}"

                elif goal_d2_now > args.goal_button_use_distance:
                    buttons, err = steering_buttons(x, y, angle, final_goal, args.invert_turn)
                    source = f"goal_priority_approach:d2={goal_d2_now:.1f}:err={err:.1f}"

                elif abs(goal_err) > args.goal_button_face_deg:
                    buttons = [0] * 8
                    if goal_err > 0:
                        buttons[2] = 1
                    else:
                        buttons[3] = 1
                    source = f"goal_priority_face:d2={goal_d2_now:.1f}:err={goal_err:.1f}"

                else:
                    goal_nudge_countdown = args.goal_button_nudge_steps
                    goal_use_hold = args.goal_use_hold_steps
                    buttons = button_vec("use")
                    source = f"goal_priority_use:d2={goal_d2_now:.1f}:err={goal_err:.1f}"

            if door_cross_countdown > 0:
                door_cross_countdown -= 1

                # Near the final button, do not keep walking into a wall.
                final_target_for_cross = waypoints[-1]
                final_d2_for_cross = dist2d(final_target_for_cross["x"], final_target_for_cross["y"], x, y)
                final_same_floor_for_cross = abs(final_target_for_cross["z"] - z) <= args.z_tolerance
                near_final_for_cross = (
                    wp_idx >= len(waypoints) - args.goal_lookback
                    and final_same_floor_for_cross
                    and final_d2_for_cross <= max(args.goal_use_radius, 90)
                )

                if near_final_for_cross and stuck:
                    door_cross_countdown = 0
                    door_cross_min_countdown = 0
                    door_lock_idx = None
                    use_hold = args.goal_use_hold_steps
                    buttons = button_vec("use")
                    source = f"final_button_stuck_use:d2={final_d2_for_cross:.1f}"

                else:
                    # Simple stable behavior: after pressing a door, force forward
                    # long enough to actually get through before route actions return.
                    if door_cross_min_countdown > 0:
                        door_cross_min_countdown -= 1

                    buttons = button_vec("move_forward")
                    source = "door_cross_forward"

                    if door_cross_countdown == 0:
                        # Once the forced crossing ends, resync to the closest future
                        # waypoint near the agent's real position.
                        start_idx = door_lock_idx if door_lock_idx is not None else wp_idx
                        start_idx = max(wp_idx, start_idx)

                        best_idx = wp_idx
                        best_d2 = float("inf")
                        hi = min(len(waypoints), start_idx + args.post_door_resync_lookahead)

                        for j in range(start_idx, hi):
                            p = waypoints[j]

                            if abs(p["z"] - z) > args.z_tolerance:
                                continue

                            cand_d2 = dist2d(p["x"], p["y"], x, y)

                            if cand_d2 < best_d2:
                                best_d2 = cand_d2
                                best_idx = j

                        if best_idx > wp_idx and best_d2 <= args.post_door_resync_radius:
                            print(
                                f"[z_breadcrumb] post_door_resync wp {wp_idx}->{best_idx} "
                                f"d2={best_d2:.1f} z={z:.1f}"
                            )
                            wp_idx = best_idx

                        door_lock_idx = None
                        door_advance_countdown = args.door_advance_after_use

            elif use_hold > 0:
                should_use = True
                use_hold -= 1
                source = "door_hold"

                if use_hold == 0:
                    door_cross_countdown = args.door_cross_steps
                    door_cross_min_countdown = min(30, args.door_cross_steps)

            elif door_lock_idx is not None:
                door_target = waypoints[min(door_lock_idx, len(waypoints) - 1)]
                lock_d2 = dist2d(door_target["x"], door_target["y"], x, y)

                err_to_door_angle = signed_angle_error(door_target["angle"], angle)
                if args.invert_turn:
                    err_to_door_angle = -err_to_door_angle

                close_enough = lock_d2 <= args.door_use_distance
                facing_enough = abs(err_to_door_angle) <= args.door_face_deg

                stuck_near_door = stuck and lock_d2 <= args.door_stuck_use_radius
                stuck_facing_door = abs(err_to_door_angle) <= args.door_stuck_use_face_deg

                if stuck_near_door and stuck_facing_door:
                    # If we slid along the wall and are close-ish/facing the door,
                    # pressing USE is better than walking into the wall forever.
                    should_use = True
                    use_hold = args.door_hold_steps
                    source = f"door_stuck_use@{door_lock_idx}:d2={lock_d2:.1f}:err={err_to_door_angle:.1f}"

                elif stuck and lock_d2 <= args.door_wall_recovery_radius:
                    # Wall-slide recovery: stop pushing straight into the wall.
                    # Back up / strafe / turn, then route logic can re-approach.
                    phase = (step // 12) % 4
                    buttons = [0] * 8

                    if phase == 0:
                        buttons[1] = 1  # move_backward
                        buttons[2] = 1  # turn_left
                        source = f"door_wall_recover_back_left@{door_lock_idx}:d2={lock_d2:.1f}"

                    elif phase == 1:
                        buttons[4] = 1  # strafe_left
                        buttons[2] = 1  # turn_left
                        source = f"door_wall_recover_strafe_left@{door_lock_idx}:d2={lock_d2:.1f}"

                    elif phase == 2:
                        buttons[1] = 1  # move_backward
                        buttons[3] = 1  # turn_right
                        source = f"door_wall_recover_back_right@{door_lock_idx}:d2={lock_d2:.1f}"

                    else:
                        buttons[5] = 1  # strafe_right
                        buttons[3] = 1  # turn_right
                        source = f"door_wall_recover_strafe_right@{door_lock_idx}:d2={lock_d2:.1f}"

                elif not close_enough:
                    buttons, err = steering_buttons(x, y, angle, door_target, args.invert_turn)
                    source = f"door_lock_approach@{door_lock_idx}:d2={lock_d2:.1f}:err={err:.1f}"

                elif not facing_enough:
                    buttons = [0] * 8
                    if err_to_door_angle > 0:
                        buttons[2] = 1
                    else:
                        buttons[3] = 1
                    source = f"door_lock_face@{door_lock_idx}:err={err_to_door_angle:.1f}"

                else:
                    should_use = True
                    use_hold = args.door_hold_steps
                    source = f"door_lock_use@{door_lock_idx}:d2={lock_d2:.1f}"

            elif current_or_route_use:
                should_use = True
                use_hold = args.door_hold_steps
                source = "door_route_action"

            if should_use:
                buttons = button_vec("use")

            # Combat sense:
            # If an enemy is centered, shoot while continuing the chosen movement.
            # If a close enemy is visible but not centered, briefly turn toward it.
            # Final goal/use awareness:
            # If we reach the end of the recorded route, treat it as an objective
            # interaction instead of just another waypoint.
            final_target = waypoints[-1]
            final_d2 = dist2d(final_target["x"], final_target["y"], x, y)
            final_same_floor = abs(final_target["z"] - z) <= args.z_tolerance
            # Do not let a bad resync/jump to the end activate goal mode from far away.
            near_route_end = (
                wp_idx >= max(0, len(waypoints) - args.goal_lookback)
                and final_same_floor
                and final_d2 <= max(args.goal_use_radius * 3, 220)
            )
            near_final_goal = final_same_floor and final_d2 <= args.goal_use_radius

            door_mode_active_for_goal = (
                door_lock_idx is not None
                or door_cross_countdown > 0
                or door_advance_countdown > 0
                or use_hold > 0
                or str(source).startswith("door_")
            )

            if not goal_done and not door_mode_active_for_goal and (near_route_end or near_final_goal):
                final_err = signed_angle_error(final_target["angle"], angle)
                if args.invert_turn:
                    final_err = -final_err

                if goal_wait_countdown > 0:
                    goal_wait_countdown -= 1
                    buttons = [0] * 8
                    source = "goal_wait_after_use"

                    if goal_wait_countdown == 0:
                        goal_done = True

                elif goal_use_hold > 0:
                    goal_use_hold -= 1
                    buttons = button_vec("use")
                    source = "goal_use_hold"

                    if goal_use_hold == 0:
                        goal_wait_countdown = args.goal_wait_after_use

                elif stuck and final_d2 <= args.goal_stuck_use_radius and abs(final_err) <= args.goal_stuck_face_deg:
                    # Final goal/button behaves like a high-priority door.
                    # If we are close, facing it, and stuck, pressing USE is better
                    # than walking into the wall forever.
                    buttons = button_vec("use")
                    goal_use_hold = args.goal_use_hold_steps
                    door_cross_countdown = 0
                    door_lock_idx = None
                    use_hold = 0
                    source = f"goal_stuck_use:d2={final_d2:.1f}:err={final_err:.1f}"

                elif stuck and final_d2 <= args.goal_wall_recover_radius:
                    # We are near the goal but sliding into a wall.
                    # Recover like a door: back/strafe/turn instead of pushing forward.
                    phase = (step // 12) % 4
                    buttons = [0] * 8

                    if phase == 0:
                        buttons[1] = 1  # move_backward
                        buttons[2] = 1  # turn_left
                        source = f"goal_wall_recover_back_left:d2={final_d2:.1f}:err={final_err:.1f}"
                    elif phase == 1:
                        buttons[4] = 1  # strafe_left
                        buttons[2] = 1  # turn_left
                        source = f"goal_wall_recover_strafe_left:d2={final_d2:.1f}:err={final_err:.1f}"
                    elif phase == 2:
                        buttons[1] = 1  # move_backward
                        buttons[3] = 1  # turn_right
                        source = f"goal_wall_recover_back_right:d2={final_d2:.1f}:err={final_err:.1f}"
                    else:
                        buttons[5] = 1  # strafe_right
                        buttons[3] = 1  # turn_right
                        source = f"goal_wall_recover_strafe_right:d2={final_d2:.1f}:err={final_err:.1f}"

                elif stuck and final_d2 <= args.goal_stuck_use_radius and abs(final_err) <= args.goal_stuck_face_deg:
                    # Final goal/button behaves like a high-priority door.
                    # If we are close, facing it, and stuck, pressing USE is better
                    # than walking into the wall forever.
                    buttons = button_vec("use")
                    goal_use_hold = args.goal_use_hold_steps
                    door_cross_countdown = 0
                    door_lock_idx = None
                    use_hold = 0
                    source = f"goal_stuck_use:d2={final_d2:.1f}:err={final_err:.1f}"

                elif stuck and final_d2 <= args.goal_wall_recover_radius:
                    # We are near the goal but sliding into a wall.
                    # Recover like a door: back/strafe/turn instead of pushing forward.
                    phase = (step // 12) % 4
                    buttons = [0] * 8

                    if phase == 0:
                        buttons[1] = 1  # move_backward
                        buttons[2] = 1  # turn_left
                        source = f"goal_wall_recover_back_left:d2={final_d2:.1f}:err={final_err:.1f}"
                    elif phase == 1:
                        buttons[4] = 1  # strafe_left
                        buttons[2] = 1  # turn_left
                        source = f"goal_wall_recover_strafe_left:d2={final_d2:.1f}:err={final_err:.1f}"
                    elif phase == 2:
                        buttons[1] = 1  # move_backward
                        buttons[3] = 1  # turn_right
                        source = f"goal_wall_recover_back_right:d2={final_d2:.1f}:err={final_err:.1f}"
                    else:
                        buttons[5] = 1  # strafe_right
                        buttons[3] = 1  # turn_right
                        source = f"goal_wall_recover_strafe_right:d2={final_d2:.1f}:err={final_err:.1f}"

                elif final_d2 > args.goal_use_radius:
                    buttons, err = steering_buttons(x, y, angle, final_target, args.invert_turn)
                    source = f"goal_approach:d2={final_d2:.1f}:err={err:.1f}"

                elif abs(final_err) > args.goal_face_deg:
                    buttons = [0] * 8
                    if final_err > 0:
                        buttons[2] = 1
                    else:
                        buttons[3] = 1
                    source = f"goal_face:err={final_err:.1f}"

                else:
                    goal_use_hold = args.goal_use_hold_steps
                    buttons = button_vec("use")
                    source = f"goal_use:d2={final_d2:.1f}"

            # Door retry safety:
            # If we recently used a door but are still stuck near that door,
            # do not let normal route walking fight the doorway. Re-lock the door.
            if (
                door_lock_idx is None
                and last_door_lock_idx is not None
                and last_door_lock_idx >= max(0, wp_idx - args.door_allow_behind_wp)
                and stuck
                and wp_idx < len(waypoints) - args.goal_lookback
            ):
                last_door = waypoints[min(last_door_lock_idx, len(waypoints) - 1)]
                last_door_d2 = dist2d(last_door["x"], last_door["y"], x, y)

                if last_door_d2 <= args.door_wall_recovery_radius:
                    door_lock_idx = last_door_lock_idx
                    use_hold = 0
                    door_cross_countdown = 0
                    source = f"door_retry_lock@{door_lock_idx}:d2={last_door_d2:.1f}"

            door_priority_active = (
                str(source).startswith("door_")
                or use_hold > 0
                or has_use_button(buttons)
            )

            # If health drops and no enemy was caught by labels, scan briefly.
            # This keeps combat route-only, but gives the agent a chance to find
            # enemies that are attacking from just outside the current view.
            if last_health_for_scan is None:
                last_health_for_scan = health
            else:
                if health < last_health_for_scan - args.damage_scan_health_drop:
                    damage_scan_countdown = args.damage_scan_steps
                last_health_for_scan = health

            if (
                damage_scan_countdown > 0
                and not str(source).startswith("door_")
                and not str(source).startswith("goal_")
                and d2 <= args.combat_route_max_d2
            ):
                damage_scan_countdown -= 1
                buttons = [0] * 8

                # Sweep left, then right, while staying on-route.
                if (damage_scan_countdown // 10) % 2 == 0:
                    buttons[2] = 1
                    source = "damage_scan_left"
                else:
                    buttons[3] = 1
                    source = "damage_scan_right"

            route_too_lost_for_combat = d2 > args.combat_route_max_d2

            if not args.disable_combat and not door_priority_active and not route_too_lost_for_combat:
                enemy = find_enemy_target(state, x, y, z, angle, args, waypoints, wp_idx)

                if enemy is None:
                    route_combat_steps = 0

                elif enemy["d2"] <= args.route_combat_distance:
                    # Route-combat mode:
                    # Do not chase enemies. If a live enemy is on/near the route,
                    # pause route motion, aim, shoot, then resume route once clear.
                    route_combat_steps += 1

                    centered = abs(enemy["err"]) <= args.route_combat_shoot_angle
                    turnable = abs(enemy["err"]) <= args.route_combat_turn_angle
                    timed_out = route_combat_steps >= args.route_combat_max_steps

                    if centered:
                        buttons = [0] * 8
                        buttons[6] = 1
                        shoot_hold = max(shoot_hold, args.combat_shoot_hold)
                        source = f"route_combat_shoot:{enemy['name']}:err={enemy['err']:.1f}:d2={enemy['d2']:.1f}"

                    elif turnable and not timed_out:
                        buttons = [0] * 8
                        if enemy["err"] > 0:
                            buttons[2] = 1
                        else:
                            buttons[3] = 1
                        source = f"route_combat_aim:{enemy['name']}:err={enemy['err']:.1f}:d2={enemy['d2']:.1f}"

                    elif timed_out:
                        # Do not get stuck fighting forever, but do not instantly
                        # abandon combat either. Fire once, then resume route logic.
                        buttons = [0] * 8
                        buttons[6] = 1
                        route_combat_steps = 0
                        source = f"route_combat_timeout_shot:{enemy['name']}:err={enemy['err']:.1f}:d2={enemy['d2']:.1f}"

                elif shoot_hold > 0:
                    buttons[6] = 1
                    shoot_hold -= 1
                    source = source + "+combat_hold"

            # Ghost-route teacher:
            # The AI gets an invisible GPS path copied from the successful human
            # route. It should follow the ghost path, not guess wall corrections.
            if args.enable_ghost_route and not str(source).startswith("route_combat_"):
                anchor_idx, anchor_d2 = ghost_anchor_index(waypoints, wp_idx, x, y, z, args)

                # Only move route progress forward.
                if anchor_idx > wp_idx:
                    wp_idx = anchor_idx

                terminal_d2 = dist2d(args.terminal_button_x, args.terminal_button_y, x, y)
                terminal_zdiff = abs(args.terminal_button_z - z)

                terminal_active = (
                    terminal_zdiff <= args.z_tolerance
                    and (
                        wp_idx >= args.terminal_min_wp
                        or terminal_d2 <= args.terminal_activation_radius
                    )
                )

                if terminal_active:
                    terminal_angle = normalize_angle(
                        math.degrees(math.atan2(args.terminal_button_y - y, args.terminal_button_x - x))
                    )
                    terminal_err = signed_angle_error(terminal_angle, angle)

                    if args.invert_turn:
                        terminal_err = -terminal_err

                    # At the end, stop walking first. Face the button, then use.
                    door_cross_countdown = 0
                    door_cross_min_countdown = 0
                    use_hold = 0

                    if terminal_d2 <= args.terminal_use_radius and abs(terminal_err) <= args.terminal_face_deg:
                        buttons = button_vec("use")
                        goal_use_hold = args.goal_use_hold_steps
                        source = f"ghost_terminal_use:d2={terminal_d2:.1f}:err={terminal_err:.1f}"

                    elif abs(terminal_err) > args.terminal_face_deg:
                        buttons = turn_only_buttons(terminal_err)
                        source = f"ghost_terminal_face:err={terminal_err:.1f}:d2={terminal_d2:.1f}"

                    else:
                        terminal_target = {
                            "x": args.terminal_button_x,
                            "y": args.terminal_button_y,
                            "z": args.terminal_button_z,
                        }
                        buttons, err = steering_buttons(x, y, angle, terminal_target, args.invert_turn)
                        source = f"ghost_terminal_approach:d2={terminal_d2:.1f}:err={err:.1f}"

                else:
                    if ghost_shoot_cooldown > 0:
                        ghost_shoot_cooldown -= 1

                    used_ghost_shoot = False

                    if ghost_shoot_cooldown <= 0:
                        shoot_idx = ghost_next_shoot_index(waypoints, wp_idx, args)

                        if shoot_idx is not None and shoot_idx <= last_ghost_shoot_idx:
                            shoot_idx = None

                        if shoot_idx is not None:
                            shoot_pt = waypoints[shoot_idx]
                            shoot_d2 = dist2d(shoot_pt["x"], shoot_pt["y"], x, y)

                            if shoot_d2 <= args.ghost_shoot_radius and abs(shoot_pt["z"] - z) <= args.z_tolerance:
                                shoot_angle = normalize_angle(
                                    math.degrees(math.atan2(shoot_pt["y"] - y, shoot_pt["x"] - x))
                                )
                                shoot_err = signed_angle_error(shoot_angle, angle)

                                if args.invert_turn:
                                    shoot_err = -shoot_err

                                if abs(shoot_err) > args.ghost_shoot_face_deg:
                                    buttons = turn_only_buttons(shoot_err)
                                    source = f"ghost_shoot_face@{shoot_idx}:err={shoot_err:.1f}:d2={shoot_d2:.1f}"
                                else:
                                    buttons = [0] * 8
                                    buttons[6] = 1
                                    ghost_shoot_cooldown = args.ghost_shoot_cooldown_steps
                                    last_ghost_shoot_idx = shoot_idx
                                    source = f"ghost_shoot_once@{shoot_idx}:err={shoot_err:.1f}:d2={shoot_d2:.1f}"

                                used_ghost_shoot = True

                    if not used_ghost_shoot:
                        if ghost_use_cooldown > 0:
                            ghost_use_cooldown -= 1

                        used_ghost_use = False

                        # After a successful USE, move forward through the door for
                        # a short window. This matches the human sequence better than
                        # pressing USE forever.
                        if ghost_cross_countdown > 0:
                            ghost_cross_countdown -= 1
                            buttons = [0] * 8
                            buttons[0] = 1
                            source = f"ghost_after_use_cross:remaining={ghost_cross_countdown}"
                            used_ghost_use = True

                        if not used_ghost_use and ghost_use_cooldown <= 0:
                            use_idx = ghost_next_use_index(waypoints, wp_idx, args)

                            # Do not repeat an already-handled use point.
                            if use_idx is not None and use_idx <= last_ghost_use_idx:
                                use_idx = None

                            if use_idx is not None:
                                use_pt = waypoints[use_idx]
                                use_d2 = dist2d(use_pt["x"], use_pt["y"], x, y)

                                if abs(use_pt["z"] - z) <= args.z_tolerance:
                                    use_angle = normalize_angle(
                                        math.degrees(math.atan2(use_pt["y"] - y, use_pt["x"] - x))
                                    )
                                    use_err = signed_angle_error(use_angle, angle)

                                    if args.invert_turn:
                                        use_err = -use_err

                                    # Too far: approach the recorded human USE spot.
                                    if use_d2 > args.ghost_use_radius:
                                        buttons, err = steering_buttons(x, y, angle, use_pt, args.invert_turn)

                                        if abs(err) > args.ghost_turn_only_deg:
                                            buttons = turn_only_buttons(err)

                                        source = f"ghost_use_approach@{use_idx}:d2={use_d2:.1f}:err={err:.1f}"
                                        used_ghost_use = True

                                    # Close but not facing: turn first, no USE spam.
                                    elif abs(use_err) > args.ghost_use_face_deg:
                                        buttons = turn_only_buttons(use_err)
                                        source = f"ghost_use_face@{use_idx}:err={use_err:.1f}:d2={use_d2:.1f}"
                                        used_ghost_use = True

                                    # Correct spot and facing: press USE briefly, then cross.
                                    else:
                                        buttons = button_vec("use")
                                        use_hold = max(use_hold, args.ghost_use_hold_steps)
                                        ghost_cross_countdown = args.ghost_after_use_cross_steps
                                        ghost_use_cooldown = args.ghost_use_cooldown_steps
                                        last_ghost_use_idx = use_idx

                                        # Move route progress beyond the handled use point.
                                        wp_idx = max(wp_idx, min(len(waypoints) - 1, use_idx + 2))

                                        source = f"ghost_use_once@{use_idx}:d2={use_d2:.1f}:err={use_err:.1f}"
                                        used_ghost_use = True

                        if not used_ghost_use:
                            target_idx = min(len(waypoints) - 1, wp_idx + args.ghost_lookahead)
                            target = waypoints[target_idx]

                            # If the target is too far because of a missed door/corner,
                            # use the anchor instead of jumping across a wall.
                            target_d2 = dist2d(target["x"], target["y"], x, y)
                            if target_d2 > args.ghost_corridor_radius:
                                target_idx = wp_idx
                                target = waypoints[target_idx]
                                target_d2 = dist2d(target["x"], target["y"], x, y)

                            buttons, err = steering_buttons(x, y, angle, target, args.invert_turn)

                            # Turn-in-place on sharp angle errors. This fixes the slow
                            # turn problem near doors/goal by not walking forward while
                            # facing the wrong way.
                            if abs(err) > args.ghost_turn_only_deg:
                                buttons = turn_only_buttons(err)

                            source = (
                                f"ghost_route@{target_idx}:anchor={anchor_idx}:"
                                f"anchor_d2={anchor_d2:.1f}:target_d2={target_d2:.1f}:err={err:.1f}"
                            )

                # Ghost route is the route teacher, so do not let stale correction
                # states force old behavior.
                if 'route_loop_count' in locals():
                    route_loop_count = 0


            # Final door/button stuck fallback:
            # If the ghost route reaches the final area but forward movement is stuck,
            # do not keep walking into the door. Press USE once, then cross.
            if args.enable_ghost_route and stuck and wp_idx >= args.ghost_final_min_wp:
                final_use_idx = ghost_next_use_index(waypoints, wp_idx, args)

                if final_use_idx is not None:
                    final_use_pt = waypoints[final_use_idx]
                    final_use_d2 = dist2d(final_use_pt["x"], final_use_pt["y"], x, y)

                    if (
                        final_use_d2 <= args.ghost_final_stuck_use_radius
                        and abs(final_use_pt["z"] - z) <= args.z_tolerance
                    ):
                        buttons = button_vec("use")
                        use_hold = max(use_hold, args.ghost_use_hold_steps)
                        ghost_cross_countdown = args.ghost_after_use_cross_steps
                        ghost_use_cooldown = args.ghost_use_cooldown_steps
                        last_ghost_use_idx = final_use_idx
                        wp_idx = max(wp_idx, min(len(waypoints) - 1, final_use_idx + 1))
                        source = f"ghost_final_stuck_use@{final_use_idx}:d2={final_use_d2:.1f}"


            # Generic stuck-door use fallback:
            # If the agent is stuck at a door line and a recorded human USE
            # marker is nearby, press USE once, then move forward through it.
            if (
                args.enable_ghost_route
                and stuck
                and ghost_cross_countdown <= 0
                and ghost_use_cooldown <= 0
            ):
                stuck_use_idx, stuck_use_d2 = ghost_nearby_use_index(
                    waypoints, wp_idx, x, y, z, args
                )

                if stuck_use_idx is not None and stuck_use_idx <= last_ghost_use_idx:
                    stuck_use_idx = None

                if stuck_use_idx is not None:
                    stuck_use_pt = waypoints[stuck_use_idx]
                    use_err, desired_angle, stuck_use_d2 = recorded_use_angle_error(
                        stuck_use_pt, x, y, angle, args
                    )

                    # Last/final doors are less forgiving. Face first, no walking.
                    face_deg = args.ghost_stuck_door_face_deg
                    if wp_idx >= args.ghost_final_door_min_wp:
                        face_deg = min(face_deg, 12)

                    if abs(use_err) > face_deg:
                        buttons = turn_only_buttons(use_err)
                        source = (
                            f"ghost_stuck_door_face@{stuck_use_idx}:"
                            f"err={use_err:.1f}:want={desired_angle:.1f}:d2={stuck_use_d2:.1f}"
                        )
                    else:
                        buttons = button_vec("use")
                        use_hold = max(use_hold, args.ghost_use_hold_steps)
                        ghost_cross_countdown = args.ghost_stuck_door_cross_steps
                        ghost_use_cooldown = args.ghost_use_cooldown_steps
                        last_ghost_use_idx = max(last_ghost_use_idx, stuck_use_idx)
                        wp_idx = max(wp_idx, min(len(waypoints) - 1, stuck_use_idx + 2))
                        source = (
                            f"ghost_stuck_door_use@{stuck_use_idx}:"
                            f"err={use_err:.1f}:want={desired_angle:.1f}:d2={stuck_use_d2:.1f}"
                        )


            # Final overshoot gate:
            # Near the final recorded use marker, if stuck or very close to the
            # last route section, stop approaching. Face the final recorded
            # human angle, press USE once, then cross/wait.
            if args.enable_ghost_route:
                final_force_wp = int(len(waypoints) * args.ghost_final_force_frac)
                final_use_idx = ghost_last_use_index(waypoints)
                final_use_pt = waypoints[final_use_idx]
                final_d2 = dist2d(final_use_pt["x"], final_use_pt["y"], x, y)

                final_active = (
                    wp_idx >= final_force_wp
                    and final_d2 <= args.ghost_final_force_radius
                    and abs(final_use_pt["z"] - z) <= args.z_tolerance
                )

                if final_active and ghost_cross_countdown <= 0:
                    final_err, final_want, final_d2 = recorded_use_angle_error(
                        final_use_pt, x, y, angle, args
                    )

                    if abs(final_err) > args.ghost_final_force_face_deg:
                        buttons = turn_only_buttons(final_err)
                        source = (
                            f"ghost_final_face@{final_use_idx}:"
                            f"err={final_err:.1f}:want={final_want:.1f}:d2={final_d2:.1f}"
                        )
                    elif ghost_use_cooldown <= 0:
                        buttons = button_vec("use")
                        use_hold = max(use_hold, args.ghost_use_hold_steps)
                        ghost_cross_countdown = args.ghost_final_force_cross_steps
                        ghost_use_cooldown = args.ghost_use_cooldown_steps
                        last_ghost_use_idx = max(last_ghost_use_idx, final_use_idx)
                        wp_idx = max(wp_idx, final_use_idx)
                        source = (
                            f"ghost_final_use@{final_use_idx}:"
                            f"err={final_err:.1f}:want={final_want:.1f}:d2={final_d2:.1f}"
                        )
                    else:
                        buttons = [0] * 8
                        source = (
                            f"ghost_final_wait@{final_use_idx}:"
                            f"cooldown={ghost_use_cooldown}:d2={final_d2:.1f}"
                        )


            # Hard final-door wall recovery:
            # The ghost route reaches the final door area, but the final point is
            # behind the wall/door. When stuck near the final wall, face the
            # crossing direction, press USE, then cross.
            final_wall_d2 = dist2d(args.final_wall_x, args.final_wall_y, x, y)
            final_wall_active = (
                args.enable_ghost_route
                and wp_idx >= int(len(waypoints) * 0.96)
                and final_wall_d2 <= args.final_wall_radius
                and z <= -100
            )

            if final_wall_active:
                cross_target = {
                    "x": args.final_cross_x,
                    "y": args.final_cross_y,
                    "z": z,
                }
                cross_buttons, cross_err = steering_buttons(
                    x, y, angle, cross_target, args.invert_turn
                )

                if final_wall_use_countdown > 0:
                    final_wall_use_countdown -= 1
                    buttons = button_vec("use")
                    source = f"final_wall_use_hold:remaining={final_wall_use_countdown}:err={cross_err:.1f}"

                elif final_wall_cross_countdown > 0:
                    final_wall_cross_countdown -= 1

                    if abs(cross_err) > args.final_cross_face_deg:
                        buttons = turn_only_buttons(cross_err)
                        source = (
                            f"final_wall_cross_face:"
                            f"remaining={final_wall_cross_countdown}:err={cross_err:.1f}"
                        )
                    else:
                        buttons = cross_buttons
                        source = (
                            f"final_wall_cross:"
                            f"remaining={final_wall_cross_countdown}:err={cross_err:.1f}"
                        )

                elif stuck or final_wall_d2 <= args.final_wall_radius:
                    if abs(cross_err) > args.final_cross_face_deg:
                        buttons = turn_only_buttons(cross_err)
                        source = f"final_wall_face:err={cross_err:.1f}:d2={final_wall_d2:.1f}"
                    else:
                        buttons = button_vec("use")
                        final_wall_use_countdown = args.final_wall_use_steps
                        final_wall_cross_countdown = args.final_wall_cross_steps
                        ghost_cross_countdown = 0
                        ghost_use_cooldown = 0
                        source = f"final_wall_use_once:err={cross_err:.1f}:d2={final_wall_d2:.1f}"


            # Terminal stuck recovery:
            # If the agent reaches the final terminal/button but freezes while
            # repeatedly pressing USE, make it self-correct instead of waiting
            # for enemy damage to bump it into place.
            terminal_recovery_active = (
                wp_idx >= int(len(waypoints) * args.terminal_stuck_min_wp_frac)
                and z <= -100
                and (
                    "ghost_terminal_use" in str(source)
                    or "door_hold" in str(source)
                    or "goal_use_hold" in str(source)
                )
            )

            if terminal_stuck_recover_countdown > 0:
                terminal_stuck_recover_countdown -= 1

                if terminal_stuck_recover_countdown > (
                    args.terminal_stuck_recover_steps - args.terminal_stuck_back_steps
                ):
                    buttons = button_vec("move_backward")
                    source = (
                        f"terminal_stuck_backoff:"
                        f"remaining={terminal_stuck_recover_countdown}"
                    )
                else:
                    terminal_target = {
                        "x": args.terminal_button_x,
                        "y": args.terminal_button_y,
                        "z": args.terminal_button_z,
                    }
                    terminal_buttons, terminal_err = steering_buttons(
                        x, y, angle, terminal_target, args.invert_turn
                    )

                    if abs(terminal_err) > args.terminal_face_deg:
                        buttons = turn_only_buttons(terminal_err)
                        source = (
                            f"terminal_stuck_reface:"
                            f"remaining={terminal_stuck_recover_countdown}:"
                            f"err={terminal_err:.1f}"
                        )
                    else:
                        buttons = terminal_buttons
                        source = (
                            f"terminal_stuck_reapproach:"
                            f"remaining={terminal_stuck_recover_countdown}:"
                            f"err={terminal_err:.1f}"
                        )

            elif terminal_recovery_active and stuck:
                terminal_stuck_recover_countdown = args.terminal_stuck_recover_steps
                buttons = button_vec("move_backward")
                source = "terminal_stuck_recovery_start"


            # Terminal magnet:
            # Once the agent is in the final route section, do not let it spam
            # USE from too far away. Pull it toward the terminal/button first,
            # then face, then use.
            terminal_target = {
                "x": args.terminal_button_x,
                "y": args.terminal_button_y,
                "z": args.terminal_button_z,
            }

            terminal_d2 = dist2d(
                args.terminal_button_x,
                args.terminal_button_y,
                x,
                y,
            )

            terminal_magnet_active = (
                wp_idx >= args.terminal_magnet_min_wp
                and z <= -100
                and y <= args.terminal_magnet_max_y
                and terminal_d2 <= args.terminal_magnet_radius
            )

            if terminal_magnet_active:
                terminal_buttons, terminal_err = steering_buttons(
                    x, y, angle, terminal_target, args.invert_turn
                )

                if terminal_use_lock_countdown > 0:
                    terminal_use_lock_countdown -= 1
                    buttons = button_vec("use")
                    source = (
                        f"terminal_use_lock:"
                        f"remaining={terminal_use_lock_countdown}:"
                        f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                    )

                elif (
                    terminal_d2 <= args.terminal_magnet_use_radius
                    and abs(terminal_err) <= args.terminal_use_lock_face_deg
                ):
                    terminal_use_lock_countdown = args.terminal_use_lock_steps
                    buttons = button_vec("use")
                    source = (
                        f"terminal_use_lock_start:"
                        f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                    )

                else:

                    if terminal_d2 > args.terminal_magnet_use_radius:
                        # Too far: gravitate to the terminal/button.
                        if abs(terminal_err) > args.terminal_magnet_face_deg * 2:
                            buttons = turn_only_buttons(terminal_err)
                            source = (
                                f"terminal_magnet_face:"
                                f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                            )
                        else:
                            buttons = terminal_buttons
                            source = (
                                f"terminal_magnet_approach:"
                                f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                            )

                    elif abs(terminal_err) > args.terminal_magnet_face_deg:
                        buttons = turn_only_buttons(terminal_err)
                        source = (
                            f"terminal_magnet_final_face:"
                            f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                        )

                    else:
                        buttons = button_vec("use")
                        source = (
                            f"terminal_magnet_use:"
                            f"d2={terminal_d2:.1f}:err={terminal_err:.1f}"
                        )

            # Final rescue door open/cross:
            # The wall rescue gets the agent to the side/final door area,
            # but if the door is still closed it must press USE before trying
            # to slide/cross into the final room.
            final_rescue_door_target = {
                "x": args.final_rescue_door_x,
                "y": args.final_rescue_door_y,
                "z": z,
            }

            final_rescue_door_d2 = dist2d(
                args.final_rescue_door_x,
                args.final_rescue_door_y,
                x,
                y,
            )

            final_rescue_door_zone = (
                wp_idx >= 254
                and z <= -100
                and y >= 1390
                and final_rescue_door_d2 <= args.final_rescue_door_radius
            )

            if final_rescue_door_cross_countdown > 0:
                final_rescue_door_cross_countdown -= 1

                try:
                    final_wall_rescue_countdown = 0
                    final_wall_cross_countdown = 0
                except Exception:
                    pass

                cross_target = {
                    "x": args.final_cross_x,
                    "y": args.final_cross_y,
                    "z": z,
                }
                cross_buttons, cross_err = steering_buttons(
                    x, y, angle, cross_target, args.invert_turn
                )
                buttons = cross_buttons
                source = (
                    f"final_rescue_door_cross:"
                    f"remaining={final_rescue_door_cross_countdown}:"
                    f"err={cross_err:.1f}"
                )

            elif final_rescue_door_use_countdown > 0:
                final_rescue_door_use_countdown -= 1

                try:
                    final_wall_rescue_countdown = 0
                    final_wall_cross_countdown = 0
                except Exception:
                    pass

                buttons = button_vec("use")
                source = (
                    f"final_rescue_door_use:"
                    f"remaining={final_rescue_door_use_countdown}:"
                    f"d2={final_rescue_door_d2:.1f}"
                )

                if final_rescue_door_use_countdown <= 0:
                    final_rescue_door_cross_countdown = args.final_rescue_door_cross_steps

            elif final_rescue_door_zone and stuck:
                final_rescue_door_use_countdown = args.final_rescue_door_use_steps

                try:
                    final_wall_rescue_countdown = 0
                    final_wall_cross_countdown = 0
                except Exception:
                    pass

                buttons = button_vec("use")
                source = (
                    f"final_rescue_door_use_start:"
                    f"d2={final_rescue_door_d2:.1f}:"
                    f"pos=({x:.1f},{y:.1f})"
                )


            # Final-wall rescue override:
            # If the controller gets stuck at the old wp=254 wall loop,
            # stop walking into the wall and force a side-door/slide recovery
            # toward the final-cross point.
            final_wall_rescue_zone = (
                wp_idx >= 254
                and z <= -100
                and y >= args.final_wall_rescue_trigger_y
                and stuck
                and "final_wall" in str(source)
            )

            if final_wall_rescue_countdown > 0:
                final_wall_rescue_countdown -= 1

                # Stop the old final_wall_cross loop from immediately
                # resetting and taking control again.
                try:
                    final_wall_cross_countdown = 0
                except Exception:
                    pass

                if final_wall_rescue_countdown > (
                    args.final_wall_rescue_steps - args.final_wall_rescue_back_steps
                ):
                    buttons = button_vec("move_backward")
                    source = (
                        f"final_wall_rescue_backoff:"
                        f"remaining={final_wall_rescue_countdown}:"
                        f"pos=({x:.1f},{y:.1f})"
                    )
                else:
                    rescue_target = {
                        "x": args.final_cross_x,
                        "y": args.final_cross_y,
                        "z": z,
                    }

                    rescue_buttons, rescue_err = steering_buttons(
                        x, y, angle, rescue_target, args.invert_turn
                    )

                    buttons = rescue_buttons
                    source = (
                        f"final_wall_rescue_slide:"
                        f"remaining={final_wall_rescue_countdown}:"
                        f"err={rescue_err:.1f}:"
                        f"target=({args.final_cross_x:.1f},{args.final_cross_y:.1f})"
                    )

            elif final_wall_rescue_zone:
                final_wall_rescue_countdown = args.final_wall_rescue_steps

                try:
                    final_wall_cross_countdown = 0
                except Exception:
                    pass

                buttons = button_vec("move_backward")
                source = (
                    f"final_wall_rescue_start:"
                    f"pos=({x:.1f},{y:.1f}):"
                    f"target=({args.final_cross_x:.1f},{args.final_cross_y:.1f})"
                )


            if ai_record_dir is not None:
                ai_record_max_wp = max(ai_record_max_wp, int(wp_idx))

                src_text = str(source)

                # Terminal/completion signal:
                # Older finish controllers used ghost_final_use/goal_use names.
                # The newer final controller uses terminal_magnet_* and
                # terminal_use_lock* names. Without this bridge, runs can reach
                # wp=260/261 near the terminal but still be discarded as
                # terminal_signal=False.
                terminal_signal_source = any(tok in src_text for tok in [
                    "ghost_terminal",
                    "ghost_final_use",
                    "goal_done",
                    "goal_wait_after_use",
                    "goal_priority_wait",
                    "final_wall_use_once",
                    "goal_use",
                    "terminal_magnet_use",
                    "terminal_use_lock",
                    "terminal_use_lock_start",
                ])

                # Conservative fallback for the exact near-finish failure mode:
                # terminal_magnet_approach/final_face at wp260 should count as
                # a terminal candidate only when already in the final waypoint
                # window and physically inside the terminal magnet radius.
                terminal_d2_now = float(locals().get("terminal_d2", 999999.0))
                near_terminal_finish_signal = (
                    int(wp_idx) >= max(0, len(waypoints) - 2)
                    and terminal_d2_now <= float(args.terminal_magnet_radius)
                    and any(tok in src_text for tok in [
                        "terminal_magnet_approach",
                        "terminal_magnet_final_face",
                        "terminal_magnet_face",
                    ])
                )

                if terminal_signal_source or near_terminal_finish_signal:
                    ai_record_terminal_signal = True

                frame_idx = len(ai_record_rows)
                frame_name = f"frame_{frame_idx:06d}.png"
                frame_rel = f"frames/{frame_name}"
                frame_path = ai_record_frames_dir / frame_name

                saved_path = _ai_record_save_frame(game, frame_path)
                if not saved_path:
                    frame_rel = ""

                tgt = target if isinstance(target, dict) else {}

                ai_record_rows.append({
                    "frame_path": frame_rel,
                    "frame": frame_idx,
                    "action": _ai_record_action_from_buttons(buttons),
                    "buttons": _ai_record_action_from_buttons(buttons),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "angle": float(angle),
                    "health": float(health),
                    "ammo": float(locals().get("ammo", 0)),
                    "kill_count": int(locals().get("kill_count", 0)),
                    "item_count": int(locals().get("item_count", 0)),
                    "done": False,
                    "truncated": False,
                    "timestamp": time.time(),
                    "source": src_text,
                    "wp_idx": int(wp_idx),
                    "target_x": float(tgt.get("x", 0.0)),
                    "target_y": float(tgt.get("y", 0.0)),
                    "target_z": float(tgt.get("z", 0.0)),
                })



            # Optional behavior-cloning student override.
            # This is intentionally late and conservative:
            # teacher/controller logic computes buttons first, then BC may override
            # only during safe non-final route movement.
            if bc_model is not None and args.bc_mix > 0:
                try:
                    state_for_bc = game.get_state()
                    screen_for_bc = getattr(state_for_bc, "screen_buffer", None) if state_for_bc is not None else None

                    if screen_for_bc is not None:
                        bc_frame_buffer.append(bc_screen_to_tensor(screen_for_bc, bc_image_size))

                    bc_can_try = (
                        len(bc_frame_buffer) >= bc_seq_len
                        and args.bc_min_wp <= wp_idx <= args.bc_max_wp
                        and bc_source_is_safe(source)
                        and bc_override_is_allowed(
                            wp_idx,
                            bc_last_override_step,
                            cooldown_steps=6,
                        )
                        and random.random() < args.bc_mix
                    )

                    if bc_can_try:
                        import torch

                        x_bc = torch.stack(list(bc_frame_buffer), dim=0).unsqueeze(0).to(next(bc_model.parameters()).device)

                        with torch.no_grad():
                            logits_bc = bc_model(x_bc)
                            probs_bc = torch.sigmoid(logits_bc)[0].detach().cpu()

                        bc_names, bc_buttons = bc_names_to_buttons(
                            bc_action_names,
                            probs_bc,
                            args.bc_threshold,
                        )

                        # Do not allow BC to replace useful teacher action with no_op.
                        if bc_names:
                            old_source = source
                            buttons = bc_buttons
                            source = (
                                "bc_policy:"
                                + "+".join(bc_names)
                                + f":old={old_source}"
                            )

                            bc_override_count += 1
                            bc_last_override_step = wp_idx
                            if args.bc_debug_every > 0 and bc_override_count % args.bc_debug_every == 0:
                                prob_text = ",".join(
                                    f"{n}={float(pv):.2f}"
                                    for n, pv in zip(bc_action_names, probs_bc)
                                )
                                print(
                                    f"[bc] override#{bc_override_count} "
                                    f"wp={wp_idx} source={source} probs={prob_text}"
                                )

                except Exception as e:
                    if args.bc_debug_every > 0:
                        print(f"[bc] disabled this tick due to error: {e}")


            # Strict final-button helper v7:
            # No-freeze final controller.
            # V6 used no_op during turn cooldown, which made the agent stand still.
            # V7 uses the real final waypoint direction and never waits with no_op in the final zone.
            try:
                near_final_wp = int(wp_idx) >= max(0, len(waypoints) - 2)
            except Exception:
                near_final_wp = False

            final_button_mode = near_final_wp or any(tag in str(source) for tag in (
                "terminal_magnet_use",
                "terminal_use_lock",
                "terminal_use_lock_start",
                "terminal_magnet_final_face",
                "terminal_magnet_face",
                "terminal_magnet_approach",
                "ghost_final_face",
                "ghost_final_use",
            ))

            if final_button_mode:
                try:
                    final_route_d2 = float(target_d2)
                except Exception:
                    final_route_d2 = 999.0

                # Compute direction to the actual final route target.
                # This is more stable than terminal_err near the button.
                try:
                    _math = __import__("math")
                    gx = float(target["x"]) - float(x)
                    gy = float(target["y"]) - float(y)
                    desired_angle = (_math.degrees(_math.atan2(gy, gx)) + 360.0) % 360.0
                    final_route_err = ((desired_angle - float(angle) + 540.0) % 360.0) - 180.0
                except Exception:
                    final_route_err = 0.0

                try:
                    final_terminal_d2 = float(locals().get("terminal_d2", final_route_d2))
                except Exception:
                    final_terminal_d2 = final_route_d2

                buttons = [0] * len(buttons)

                if final_turn_pulse_cooldown > 0:
                    final_turn_pulse_cooldown -= 1

                use_distance = 8.0
                align_deg = 14.0

                # Close enough: do not rotate anymore. Press USE, then tiny nudge forward if needed.
                if final_route_d2 <= use_distance:
                    if final_use_nudge_countdown > 0 and final_route_d2 > 2.0:
                        if len(buttons) >= 1:
                            buttons[0] = 1
                        final_use_nudge_countdown -= 1
                        source = (
                            "strict_final_micro_nudge_v7:"
                            + f"remaining={final_use_nudge_countdown}:route_d2={final_route_d2:.1f}:route_err={final_route_err:.1f}:td2={final_terminal_d2:.1f}:"
                            + str(source)
                        )
                    else:
                        if len(buttons) >= 8:
                            buttons[7] = 1
                        final_use_attempt_count += 1
                        final_use_nudge_countdown = 2
                        final_last_use_step = step
                        source = (
                            "strict_final_use_v7:"
                            + f"attempt={final_use_attempt_count}:route_d2={final_route_d2:.1f}:route_err={final_route_err:.1f}:td2={final_terminal_d2:.1f}:"
                            + str(source)
                        )

                # Not close: one-tick turn pulse, then forward movement. Never no_op.
                elif abs(final_route_err) > align_deg and final_turn_pulse_cooldown == 0:
                    if final_route_err > 0 and len(buttons) >= 3:
                        buttons[2] = 1
                        final_turn_pulse_cooldown = 1
                        source = (
                            "strict_final_pulse_left_v7:"
                            + f"cooldown={final_turn_pulse_cooldown}:route_d2={final_route_d2:.1f}:route_err={final_route_err:.1f}:td2={final_terminal_d2:.1f}:"
                            + str(source)
                        )
                    elif final_route_err < 0 and len(buttons) >= 4:
                        buttons[3] = 1
                        final_turn_pulse_cooldown = 1
                        source = (
                            "strict_final_pulse_right_v7:"
                            + f"cooldown={final_turn_pulse_cooldown}:route_d2={final_route_d2:.1f}:route_err={final_route_err:.1f}:td2={final_terminal_d2:.1f}:"
                            + str(source)
                        )

                else:
                    # During cooldown or when aligned, move forward instead of freezing.
                    if len(buttons) >= 1:
                        buttons[0] = 1
                    source = (
                        "strict_final_forward_v7:"
                        + f"cooldown={final_turn_pulse_cooldown}:route_d2={final_route_d2:.1f}:route_err={final_route_err:.1f}:td2={final_terminal_d2:.1f}:"
                        + str(source)
                    )

            reward = game.make_action(buttons, 1)

            this_action_name = action_name(buttons)
            if game.is_episode_finished():
                ai_record_true_episode_finished = True
                ai_record_final_action_name = this_action_name
                ai_record_final_action_used = ("use" in this_action_name.split("+"))


            if step % 25 == 0:
                print(
                    f"[z_breadcrumb] step={step} src={source} "
                    f"wp={wp_idx}/{len(waypoints)} d2={target_d2:.1f} d3={target_d3:.1f} "
                    f"z={z:.1f} target_z={target['z']:.1f} zdiff={z_diff:.1f} "
                    f"stuck={stuck} active={action_name(buttons)} "
                    f"pos=({x:.1f},{y:.1f},{z:.1f}) "
                    f"target=({target['x']:.1f},{target['y']:.1f},{target['z']:.1f}) "
                    f"angle={angle:.1f} health={health:.1f} reward={reward}"
                )

            step += 1
            time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("[z_breadcrumb] stopped by user")

    finally:
        game.close()


    if ai_record_dir is not None:
        final_health = float(locals().get("health", 0.0))
        total_wp = len(waypoints)
        min_success_wp = int(total_wp * args.ai_success_min_wp_frac)

        # TERMINAL SIGNAL SUCCESS RULE:
        # User goal: if the agent reaches/triggers the terminal signal while alive,
        # save the episode as success. Waypoint max and VizDoom finish flags are diagnostic only.
        ai_success = (
            final_health > 0
            and ai_record_terminal_signal
        )
        # TERMINAL SIGNAL FORCE-SUCCESS OVERRIDE:
        # User goal: once terminal_signal=True is reached while alive,
        # save the run as success even if max_wp/true_finished/final_action flags disagree.
        try:
            ai_record_terminal_signal_forced_success = False
            if final_health > 0 and ai_record_terminal_signal:
                ai_success = True
                ai_record_terminal_signal_forced_success = True
        except Exception:
            ai_record_terminal_signal_forced_success = False

        if ai_success:
            _ai_record_write_csv(ai_record_dir / "actions.csv", ai_record_rows)
            (ai_record_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2))

            final_dir = ai_record_dir.parent / ai_record_dir.name.replace(
                "session_ai_tmp_", "session_ai_success_"
            )
            ai_record_dir.rename(final_dir)
            print(f"[ai_record] SUCCESS saved (TERMINAL SIGNAL): {final_dir} "
                  f"terminal_signal={ai_record_terminal_signal} "
                  f"health={final_health} "
                  f"max_wp={ai_record_max_wp}/{len(waypoints)}")
        else:
            print(
                "[ai_record] not success; "
                f"max_wp={ai_record_max_wp}/{total_wp} "
                f"health={final_health} "
                f"terminal_signal={ai_record_terminal_signal}"
            )
            shutil.rmtree(ai_record_dir, ignore_errors=True)


    print("[z_breadcrumb] done")


if __name__ == "__main__":
    main()
