#!/usr/bin/env python3
import argparse
import json
import math
import time
from collections import deque
from pathlib import Path


AMMO_KEYS = ("ammo_clip", "ammo_shell", "ammo_cell", "ammo_misl")


def read_json(path):
    try:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def write_json_atomic(path, data):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(p)


def load_route(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "x" in obj and "y" in obj:
            pts.append(obj)
    if not pts:
        raise SystemExit(f"[error] no x/y route points in {path}")
    return pts


def ammo_total(s):
    return sum(int(s.get(k, 0) or 0) for k in AMMO_KEYS)


def dist_xy(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def wrap180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def heading_to(a, b):
    dx = float(b["x"]) - float(a["x"])
    dy = float(b["y"]) - float(a["y"])
    return math.degrees(math.atan2(dy, dx))


def nearest_index(route, s):
    best_i = 0
    best_d = float("inf")
    for i, pt in enumerate(route):
        d = dist_xy(s, pt)
        if d < best_d:
            best_i = i
            best_d = d
    return best_i, best_d


def same_state(a, b):
    if not a or not b:
        return False

    try:
        moved = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
    except Exception:
        moved = 999999.0

    angle_delta = abs(wrap180(float(a.get("angle", 0.0)) - float(b.get("angle", 0.0))))

    return (
        moved < 0.10
        and angle_delta < 0.50
        and int(a.get("health", 0)) == int(b.get("health", 0))
        and int(a.get("armor", 0)) == int(b.get("armor", 0))
        and int(a.get("kills", 0)) == int(b.get("kills", 0))
        and ammo_total(a) == ammo_total(b)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    ap.add_argument("--out", default="/tmp/doomretro_ai_advice.json")
    ap.add_argument("--hz", type=float, default=10.0)

    # Route watcher.
    ap.add_argument("--lookahead", type=int, default=22)
    ap.add_argument("--offroute", type=float, default=160.0)
    ap.add_argument("--final-zone-distance", type=float, default=18.0)
    ap.add_argument("--no-progress-lock", dest="progress_lock", action="store_false", default=True)
    ap.add_argument("--backtrack-window", type=int, default=2)
    ap.add_argument("--search-ahead", type=int, default=40)
    ap.add_argument("--route-reset-distance", type=float, default=128.0)
    ap.add_argument("--route-reset-after-wp", type=int, default=10)

    # Heading watcher.
    ap.add_argument("--turn-deadzone", type=float, default=30.0)
    ap.add_argument("--turn-release", type=float, default=14.0)
    ap.add_argument("--turn-smoothing", type=float, default=0.96)
    ap.add_argument("--recovery-distance", type=float, default=9999.0)
    ap.add_argument("--recovery-angle-limit", type=float, default=35.0)

    # Stuck / idle watcher.
    ap.add_argument("--stuck-window", type=float, default=3.0)
    ap.add_argument("--stuck-min-dist", type=float, default=8.0)
    ap.add_argument("--idle-seconds", type=float, default=2.0)

    # Survival/ammo watcher.
    ap.add_argument("--low-health", type=int, default=45)
    ap.add_argument("--critical-health", type=int, default=25)
    ap.add_argument("--low-ammo", type=int, default=12)
    ap.add_argument("--critical-ammo", type=int, default=4)

    ap.add_argument("--print-every", type=float, default=0.5)
    ap.add_argument("--max-state-age", type=float, default=2.0)
    ap.add_argument("--require-start", action="store_true")
    ap.add_argument("--start-distance", type=float, default=96.0)
    ap.add_argument("--start-confirm", type=float, default=0.5)

    args = ap.parse_args()

    route = load_route(args.route)
    final_wp = route[-1]

    print(f"[watchers] route: {args.route}")
    print(f"[watchers] points: {len(route)}")
    print(f"[watchers] state: {args.state_json}")
    print(f"[watchers] out:   {args.out}")

    last_wp = 0
    prev_state = None
    unchanged_since = None
    history = deque()
    last_print = 0.0

    smoothed_turn_error = None
    stable_route_advice = "move_forward"
    run_started = False
    near_start_since = None

    frame_dt = 1.0 / max(args.hz, 1.0)

    try:
        while True:
            loop_start = time.time()
            s = read_json(args.state_json)

            if not s or "x" not in s or "y" not in s:
                time.sleep(frame_dt)
                continue

            now = time.time()

            # Guard against stale state JSON when the shm reader is stopped.
            try:
                state_age = now - Path(args.state_json).stat().st_mtime
            except Exception:
                state_age = 999999.0

            if state_age > args.max_state_age:
                stale_advice = {
                    "timestamp": now,
                    "source": "doomretro_all_watchers_v2_phase1_bridge",
                    "route": {
                        "status": "STALE_STATE",
                        "advice": "wait_for_live_state",
                    },
                    "stuck": {
                        "stuck": False,
                        "idle_or_paused": True,
                    },
                    "combat": {
                        "status": "unknown_stale_state",
                        "shoot_permission": "hold_fire",
                    },
                    "terminal": {
                        "alive": "unknown",
                        "route_final_zone_reached": False,
                        "true_level_success": "unknown_stale_state",
                    },
                    "watchers_enabled": ["stale_state_guard"],
                    "state_age_seconds": state_age,
                }
                write_json_atomic(args.out, stale_advice)

                if now - last_print >= args.print_every:
                    print(
                        f"[watchers] STALE_STATE age={state_age:.2f}s "
                        f"advice=wait_for_live_state",
                        flush=True,
                    )
                    last_print = now

                time.sleep(frame_dt)
                prev_state = None
                continue

            # -------------------------
            # Route progress watcher.
            # -------------------------
            raw_i, raw_route_dist = nearest_index(route, s)

            start_dist = dist_xy(s, route[0])

            # Optional run-start gate. This prevents the watcher from starting
            # from an old final-zone state if the game was not reset.
            if args.require_start and not run_started:
                if start_dist <= args.start_distance:
                    if near_start_since is None:
                        near_start_since = now
                    if now - near_start_since >= args.start_confirm:
                        run_started = True
                        last_wp = 0
                        smoothed_turn_error = None
                        stable_route_advice = "move_forward"
                        print(
                            f"[watchers] RUN_STARTED start_dist={start_dist:.1f}",
                            flush=True,
                        )
                else:
                    near_start_since = None

                if not run_started:
                    wait_advice = {
                        "timestamp": now,
                        "source": "doomretro_all_watchers_v2_phase1_bridge",
                        "route": {
                            "status": "WAITING_FOR_ROUTE_START",
                            "advice": "reset_or_move_to_start",
                            "start_dist": start_dist,
                            "start_distance_required": args.start_distance,
                            "raw_waypoint": raw_i,
                            "total_waypoints": len(route) - 1,
                        },
                        "stuck": {
                            "stuck": False,
                            "idle_or_paused": True,
                        },
                        "combat": {
                            "status": "quiet",
                            "shoot_permission": "hold_fire",
                        },
                        "terminal": {
                            "alive": bool(s.get("alive", True)),
                            "route_final_zone_reached": False,
                            "true_level_success": "not_started",
                        },
                        "watchers_enabled": ["start_gate"],
                        "state": {
                            "x": float(s["x"]),
                            "y": float(s["y"]),
                            "angle": float(s.get("angle", 0.0)),
                            "health": int(s.get("health", 0)),
                            "kills": int(s.get("kills", 0)),
                            "ammo": ammo_total(s),
                        },
                    }
                    write_json_atomic(args.out, wait_advice)

                    if now - last_print >= args.print_every:
                        print(
                            f"[watchers] WAITING_FOR_ROUTE_START "
                            f"start_dist={start_dist:.1f} raw_wp={raw_i}",
                            flush=True,
                        )
                        last_print = now

                    prev_state = dict(s)
                    time.sleep(frame_dt)
                    continue
            if (
                args.progress_lock
                and last_wp >= args.route_reset_after_wp
                and raw_i <= args.backtrack_window
                and start_dist <= args.route_reset_distance
            ):
                print(
                    f"[watchers] ROUTE_PROGRESS_RESET last_wp={last_wp} "
                    f"raw_wp={raw_i} start_dist={start_dist:.1f}",
                    flush=True,
                )
                last_wp = 0
                smoothed_turn_error = None
                stable_route_advice = "move_forward"

            if args.progress_lock:
                min_allowed = max(0, last_wp - args.backtrack_window)
                max_allowed = min(len(route) - 1, last_wp + args.search_ahead)
                i = max(min_allowed, min(raw_i, max_allowed))
                if i > last_wp:
                    last_wp = i
                route_dist = dist_xy(s, route[i])
            else:
                i = raw_i
                route_dist = raw_route_dist

            target_i = min(i + args.lookahead, len(route) - 1)
            target = route[target_i]
            final_dist = dist_xy(s, final_wp)
            final_zone_reached = final_dist <= args.final_zone_distance or i >= len(route) - 2

            # -------------------------
            # Heading watcher.
            # -------------------------
            route_base = route[i]
            route_forward = route[min(i + args.lookahead, len(route) - 1)]

            route_heading = heading_to(route_base, route_forward)
            point_heading = heading_to(s, target)

            if route_dist > args.recovery_distance:
                target_heading = point_heading
                heading_source = "recovery_point"
            else:
                target_heading = route_heading
                heading_source = "route_tangent"

            angle = float(s.get("angle", 0.0))
            raw_turn_error = wrap180(target_heading - angle)

            if route_dist <= args.recovery_distance and abs(raw_turn_error) > args.recovery_angle_limit:
                raw_turn_error = math.copysign(args.recovery_angle_limit, raw_turn_error)

            if smoothed_turn_error is None:
                smoothed_turn_error = raw_turn_error
            else:
                delta = wrap180(raw_turn_error - smoothed_turn_error)
                smoothed_turn_error = wrap180(
                    smoothed_turn_error + (1.0 - args.turn_smoothing) * delta
                )

            turn_error = smoothed_turn_error

            if abs(turn_error) >= 160.0 and stable_route_advice in ("turn_left", "turn_right"):
                pass
            elif turn_error > args.turn_deadzone:
                stable_route_advice = "turn_left"
            elif turn_error < -args.turn_deadzone:
                stable_route_advice = "turn_right"
            elif abs(turn_error) <= args.turn_release:
                stable_route_advice = "move_forward"

            if final_zone_reached:
                route_status = "FINAL_ZONE_REACHED"
                route_advice = "finish_or_use"
            elif route_dist > args.offroute:
                route_status = "OFF_ROUTE"
                route_advice = stable_route_advice
            else:
                route_status = "ON_ROUTE"
                route_advice = stable_route_advice

            # -------------------------
            # Stuck / idle watcher.
            # -------------------------
            history.append((now, float(s["x"]), float(s["y"])))
            while history and now - history[0][0] > args.stuck_window:
                history.popleft()

            if prev_state is None or not same_state(s, prev_state):
                unchanged_since = None
            elif unchanged_since is None:
                unchanged_since = now

            unchanged_seconds = 0.0 if unchanged_since is None else now - unchanged_since
            idle_or_paused = unchanged_seconds >= args.idle_seconds

            stuck_dist = 999999.0
            if len(history) >= 2:
                oldest = {"x": history[0][1], "y": history[0][2]}
                newest = {"x": history[-1][1], "y": history[-1][2]}
                stuck_dist = dist_xy(oldest, newest)

            stuck = (
                stuck_dist < args.stuck_min_dist
                and not final_zone_reached
                and not idle_or_paused
            )

            # -------------------------
            # Combat watcher, Phase 1.
            # This is inference only until bridge exposes visible enemies.
            # -------------------------
            health = int(s.get("health", 0))
            armor = int(s.get("armor", 0))
            kills = int(s.get("kills", 0))
            ammo = ammo_total(s)
            alive = bool(s.get("alive", health > 0)) and health > 0

            prev_health = health if prev_state is None else int(prev_state.get("health", health))
            prev_kills = kills if prev_state is None else int(prev_state.get("kills", kills))
            prev_ammo = ammo if prev_state is None else ammo_total(prev_state)

            health_delta = health - prev_health
            kills_delta = kills - prev_kills
            ammo_delta = ammo - prev_ammo

            if not alive:
                combat_status = "dead"
            elif kills_delta > 0:
                combat_status = "enemy_killed"
            elif health_delta < 0:
                combat_status = "taking_damage"
            elif ammo_delta < 0:
                combat_status = "fired_weapon"
            else:
                combat_status = "quiet"

            enemy_visible = "unknown_until_bridge_phase2"
            enemy_in_crosshair = "unknown_until_bridge_phase2"

            shoot_permission = "hold_fire"
            if enemy_in_crosshair is True and ammo > 0:
                shoot_permission = "shoot"
            elif combat_status == "taking_damage" and ammo > 0:
                shoot_permission = "defensive_fire_allowed"
            elif ammo <= 0:
                shoot_permission = "no_ammo"

            # -------------------------
            # Ammo watcher.
            # -------------------------
            ammo_low = ammo <= args.low_ammo
            ammo_critical = ammo <= args.critical_ammo

            # -------------------------
            # Survival watcher.
            # -------------------------
            if not alive:
                survival_status = "dead"
            elif health <= args.critical_health:
                survival_status = "critical"
            elif health <= args.low_health:
                survival_status = "low"
            else:
                survival_status = "healthy"

            # -------------------------
            # Door/use/final watcher.
            # Door detection is route/final-zone only until bridge phase2.
            # -------------------------
            use_recommended = route_advice == "finish_or_use"
            door_or_switch_visible = "unknown_until_bridge_phase2"

            # -------------------------
            # Terminal watcher.
            # True level complete requires bridge phase2 gamestate/intermission.
            # -------------------------
            terminal = {
                "alive": alive,
                "route_final_zone_reached": final_zone_reached,
                "true_level_success": "unknown_until_bridge_phase2",
                "needs_bridge_fields": [
                    "gamestate",
                    "intermission",
                    "level_complete",
                    "playerstate",
                ],
            }

            advice = {
                "timestamp": now,
                "source": "doomretro_all_watchers_v2_phase1_bridge",
                "route": {
                    "status": route_status,
                    "advice": route_advice,
                    "waypoint": i,
                    "target_waypoint": target_i,
                    "last_progress_waypoint": last_wp,
                    "total_waypoints": len(route) - 1,
                    "route_dist": route_dist,
                    "final_dist": final_dist,
                    "final_zone_reached": final_zone_reached,
                    "progress_ratio": i / max(1, len(route) - 1),
                    "target_heading": target_heading,
                    "heading_source": heading_source,
                    "raw_turn_error": raw_turn_error,
                    "turn_error": turn_error,
                    "angle": angle,
                },
                "stuck": {
                    "stuck": stuck,
                    "window_dist": stuck_dist,
                    "window_seconds": args.stuck_window,
                    "idle_or_paused": idle_or_paused,
                    "unchanged_seconds": unchanged_seconds,
                },
                "combat": {
                    "status": combat_status,
                    "health_delta": health_delta,
                    "ammo_delta": ammo_delta,
                    "kills_delta": kills_delta,
                    "enemy_visible": enemy_visible,
                    "enemy_in_crosshair": enemy_in_crosshair,
                    "shoot_permission": shoot_permission,
                    "phase": "phase1_inference_only",
                },
                "ammo": {
                    "total": ammo,
                    "clip": int(s.get("ammo_clip", 0)),
                    "shell": int(s.get("ammo_shell", 0)),
                    "cell": int(s.get("ammo_cell", 0)),
                    "misl": int(s.get("ammo_misl", 0)),
                    "delta": ammo_delta,
                    "low": ammo_low,
                    "critical": ammo_critical,
                },
                "survival": {
                    "health": health,
                    "armor": armor,
                    "alive": alive,
                    "status": survival_status,
                },
                "door_use": {
                    "use_recommended": use_recommended,
                    "door_or_switch_visible": door_or_switch_visible,
                    "reason": "final_zone_route_logic_only_until_bridge_phase2",
                },
                "terminal": terminal,
                "watchers_enabled": [
                    "route_progress",
                    "route_recovery",
                    "heading_smoothing",
                    "final_zone",
                    "stuck",
                    "idle_or_paused",
                    "combat_inference",
                    "ammo",
                    "survival",
                    "door_use_placeholder",
                    "terminal_placeholder",
                    "enemy_visibility_placeholder",
                ],
                "state": {
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                    "angle": angle,
                    "health": health,
                    "armor": armor,
                    "kills": kills,
                    "ammo": ammo,
                    "alive": alive,
                },
            }

            write_json_atomic(args.out, advice)

            if now - last_print >= args.print_every:
                print(
                    f"[watchers] {route_status} "
                    f"wp={i}/{len(route)-1} "
                    f"advice={route_advice} "
                    f"err={turn_error:6.1f} "
                    f"src={heading_source} "
                    f"final={final_dist:6.1f} "
                    f"combat={combat_status} "
                    f"shoot={shoot_permission} "
                    f"health={health} ammo={ammo} "
                    f"stuck={stuck} idle={idle_or_paused}",
                    flush=True,
                )
                last_print = now

            prev_state = dict(s)

            elapsed = time.time() - loop_start
            sleep_time = frame_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[watchers] stopped.")


if __name__ == "__main__":
    main()
