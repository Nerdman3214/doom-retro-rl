#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import argparse
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


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    # Order matters.
    for name in ["POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "HEALTH"]:
        game.add_available_game_variable(getattr(vzd.GameVariable, name))

    return game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-session", default="longest_z")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--advance-dist", type=float, default=35)
    ap.add_argument("--offroute-dist", type=float, default=90)
    ap.add_argument("--z-tolerance", type=float, default=24)
    ap.add_argument("--z-weight", type=float, default=2.5)
    ap.add_argument("--z-resync-radius", type=float, default=260)
    ap.add_argument("--max-z-resync-lookahead", type=int, default=120)
    ap.add_argument("--stuck-steps", type=int, default=18)
    ap.add_argument("--stuck-min-dist", type=float, default=6)
    ap.add_argument("--door-lookahead", type=int, default=90)
    ap.add_argument("--door-radius", type=float, default=180)
    ap.add_argument("--door-use-distance", type=float, default=42)
    ap.add_argument("--door-face-deg", type=float, default=25)
    ap.add_argument("--door-hold-steps", type=int, default=18)
    ap.add_argument("--door-advance-after-use", type=int, default=8)
    ap.add_argument("--door-stuck-only", action="store_true")
    ap.add_argument("--route-action-radius", type=float, default=110)
    ap.add_argument("--episode-timeout", type=int, default=8000)
    ap.add_argument("--invert-turn", action="store_true")
    args = ap.parse_args()

    route_session = route_session_from_arg(args.route_session)
    raw, waypoints = load_route(route_session, args.frame_stride)

    print("[z_breadcrumb] route_session:", route_session)
    print("[z_breadcrumb] raw frames:", len(raw))
    print("[z_breadcrumb] waypoints:", len(waypoints))
    print("[z_breadcrumb] z range:", min(p["z"] for p in raw), "to", max(p["z"] for p in raw))

    game = make_game(args)
    game.init()
    game.new_episode()

    wp_idx = 0
    step = 0
    use_hold = 0
    door_advance_countdown = 0
    pos_hist = deque(maxlen=args.stuck_steps)
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

            # Door/use awareness:
            # A door needs an approach sequence, not only repeated USE.
            # Move toward the use point, face it, then hold USE.
            use_idx, use_d2 = find_upcoming_use(waypoints, wp_idx, x, y, z, args)

            near_upcoming_use = (
                use_idx is not None
                and use_d2 <= args.door_radius
            )

            current_or_route_use = has_use_button(buttons)

            if door_advance_countdown > 0:
                door_advance_countdown -= 1
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                source = "door_force_advance"

            should_use = False

            if use_hold > 0:
                should_use = True
                use_hold -= 1
                source = "door_hold"

                if use_hold == 0:
                    door_advance_countdown = args.door_advance_after_use

            elif near_upcoming_use and (stuck or not args.door_stuck_only):
                door_target = waypoints[use_idx]

                # Use angle from the human route at the door when available.
                err_to_door_angle = signed_angle_error(door_target["angle"], angle)
                if args.invert_turn:
                    err_to_door_angle = -err_to_door_angle

                close_enough = use_d2 <= args.door_use_distance
                facing_enough = abs(err_to_door_angle) <= args.door_face_deg

                if not close_enough:
                    # Approach the door/use point first.
                    buttons, err = steering_buttons(x, y, angle, door_target, args.invert_turn)
                    source = f"door_approach@{use_idx}:d2={use_d2:.1f}:err={err:.1f}"

                elif not facing_enough:
                    # Face the direction Steven faced when pressing use.
                    buttons = [0] * 8
                    if err_to_door_angle > 0:
                        buttons[2] = 1
                    else:
                        buttons[3] = 1
                    source = f"door_face@{use_idx}:err={err_to_door_angle:.1f}"

                else:
                    should_use = True
                    use_hold = args.door_hold_steps
                    source = f"door_use@{use_idx}:d2={use_d2:.1f}"

            elif current_or_route_use:
                should_use = True
                use_hold = args.door_hold_steps
                source = "door_route_action"

            if should_use:
                buttons = button_vec("use")

            reward = game.make_action(buttons, 1)

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

    print("[z_breadcrumb] done")


if __name__ == "__main__":
    main()
