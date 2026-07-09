#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
        f.write("\n")
    os.replace(tmp, path)

def load_route(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def dist_xy(a, b):
    return math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))

def wrap180(a):
    return ((a + 180.0) % 360.0) - 180.0

def heading_to(a, b):
    dx = float(b["x"]) - float(a["x"])
    dy = float(b["y"]) - float(a["y"])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(a.get("angle", 0.0))
    return math.degrees(math.atan2(dy, dx)) % 360.0

def nearest_index(route, s):
    best_i = 0
    best_d = 1e9
    for i, p in enumerate(route):
        d = math.hypot(float(p["x"]) - float(s["x"]), float(p["y"]) - float(s["y"]))
        if d < best_d:
            best_i = i
            best_d = d
    return best_i, best_d

def ammo_total(s):
    return (
        int(s.get("ammo_clip", 0))
        + int(s.get("ammo_shell", 0))
        + int(s.get("ammo_cell", 0))
        + int(s.get("ammo_misl", 0))
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", required=True)
    p.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    p.add_argument("--out", default="/tmp/doomretro_ai_advice.json")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--lookahead", type=int, default=6)
    p.add_argument("--offroute", type=float, default=80.0)
    p.add_argument("--final-zone-distance", type=float, default=18.0)
    p.add_argument("--no-progress-lock", dest="progress_lock", action="store_false", default=True)
    p.add_argument("--backtrack-window", type=int, default=3)
    p.add_argument("--search-ahead", type=int, default=35)
    p.add_argument("--stuck-window", type=float, default=3.0)
    p.add_argument("--stuck-min-dist", type=float, default=8.0)
    p.add_argument("--print-every", type=float, default=0.5)
    args = p.parse_args()

    route = load_route(args.route)
    if not route:
        raise SystemExit("[error] route is empty")

    final_wp = route[-1]
    last_wp = 0
    history = deque()
    prev = None
    last_print = 0.0
    unchanged_since = None

    print("[watchers] route:", args.route)
    print("[watchers] points:", len(route))
    print("[watchers] state:", args.state_json)
    print("[watchers] out:  ", args.out)

    dt = 1.0 / max(args.hz, 1.0)

    try:
        while True:
            now = time.time()
            s = read_json(args.state_json)
            if not s or "x" not in s or "y" not in s:
                time.sleep(dt)
                continue

            raw_i, raw_route_dist = nearest_index(route, s)

            if args.progress_lock:
                min_allowed = max(0, last_wp - args.backtrack_window)
                max_allowed = min(len(route) - 1, last_wp + args.search_ahead)

                if raw_i < min_allowed:
                    i = min_allowed
                elif raw_i > max_allowed:
                    i = max_allowed
                else:
                    i = raw_i

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

            target_heading = heading_to(s, target)
            angle = float(s.get("angle", 0.0))
            turn_error = wrap180(target_heading - angle)

            if final_zone_reached:
                route_status = "FINAL_ZONE_REACHED"
                route_advice = "finish_or_use"
            elif route_dist > args.offroute:
                route_status = "OFF_ROUTE"
                route_advice = "turn_left" if turn_error > 8 else "turn_right" if turn_error < -8 else "move_forward"
            else:
                route_status = "ON_ROUTE"
                route_advice = "turn_left" if turn_error > 8 else "turn_right" if turn_error < -8 else "move_forward"

            health = int(s.get("health", 0))
            armor = int(s.get("armor", 0))
            kills = int(s.get("kills", 0))
            ammo = ammo_total(s)

            if prev is None:
                health_delta = 0
                armor_delta = 0
                ammo_delta = 0
                kills_delta = 0
            else:
                health_delta = health - int(prev.get("health", 0))
                armor_delta = armor - int(prev.get("armor", 0))
                ammo_delta = ammo - ammo_total(prev)
                kills_delta = kills - int(prev.get("kills", 0))

            combat_status = "quiet"
            if health_delta < 0:
                combat_status = "taking_damage"
            elif kills_delta > 0:
                combat_status = "enemy_killed"
            elif ammo_delta < 0:
                combat_status = "fired_weapon"

            survival_status = "healthy"
            if health <= 25:
                survival_status = "critical_health"
            elif health <= 50:
                survival_status = "low_health"

            ammo_status = "ammo_ok"
            if ammo <= 0:
                ammo_status = "out_of_ammo"
            elif ammo <= 10:
                ammo_status = "low_ammo"

            history.append((now, float(s["x"]), float(s["y"])))
            while history and now - history[0][0] > args.stuck_window:
                history.popleft()

            if prev is None:
                unchanged_since = now
            else:
                same_state = (
                    abs(float(s["x"]) - float(prev.get("x", s["x"]))) < 0.01
                    and abs(float(s["y"]) - float(prev.get("y", s["y"]))) < 0.01
                    and abs(float(s.get("angle", 0.0)) - float(prev.get("angle", s.get("angle", 0.0)))) < 0.01
                    and health_delta == 0
                    and armor_delta == 0
                    and ammo_delta == 0
                    and kills_delta == 0
                )
                if same_state:
                    if unchanged_since is None:
                        unchanged_since = now
                else:
                    unchanged_since = now

            unchanged_seconds = 0.0 if unchanged_since is None else now - unchanged_since
            idle_or_paused = unchanged_seconds >= 2.0

            stuck = False
            stuck_dist = 0.0
            if len(history) >= 2:
                oldest = {"x": history[0][1], "y": history[0][2]}
                newest = {"x": history[-1][1], "y": history[-1][2]}
                stuck_dist = dist_xy(oldest, newest)

                # Exact no-change is often pause/terminal-checking idle, not real stuck.
                stuck = (
                    stuck_dist < args.stuck_min_dist
                    and not final_zone_reached
                    and not idle_or_paused
                )

            advice = {
                "timestamp": now,
                "route": {
                    "status": route_status,
                    "advice": route_advice,
                    "waypoint": i,
                    "target_waypoint": target_i,
                    "total_waypoints": len(route),
                    "route_dist": route_dist,
                    "final_dist": final_dist,
                    "target_heading": target_heading,
                    "turn_error": turn_error,
                    "final_zone_reached": final_zone_reached,
                },
                "combat": {
                    "status": combat_status,
                    "health_delta": health_delta,
                    "ammo_delta": ammo_delta,
                    "kills_delta": kills_delta,
                },
                "survival": {
                    "status": survival_status,
                    "health": health,
                    "armor": armor,
                    "alive": bool(s.get("alive", False)),
                },
                "ammo": {
                    "status": ammo_status,
                    "total": ammo,
                    "clip": int(s.get("ammo_clip", 0)),
                    "shell": int(s.get("ammo_shell", 0)),
                    "cell": int(s.get("ammo_cell", 0)),
                    "misl": int(s.get("ammo_misl", 0)),
                },
                "stuck": {
                    "stuck": stuck,
                    "window_dist": stuck_dist,
                    "window_seconds": args.stuck_window,
                    "idle_or_paused": idle_or_paused,
                    "unchanged_seconds": unchanged_seconds,
                },
                "terminal": {
                    "true_level_success": "unknown_until_c_bridge_adds_gamestate",
                    "route_final_zone_reached": final_zone_reached,
                },
                "state": {
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                    "angle": angle,
                },
            }

            write_json_atomic(args.out, advice)

            if now - last_print >= args.print_every:
                print(
                    f"[watchers] {route_status} wp={i}/{len(route)-1} "
                    f"advice={route_advice} err={turn_error:6.1f} "
                    f"final={final_dist:6.1f} combat={combat_status} "
                    f"health={health} ammo={ammo} stuck={stuck} idle={idle_or_paused}",
                    flush=True,
                )
                last_print = now

            prev = dict(s)
            time.sleep(dt)

    except KeyboardInterrupt:
        print()
        print("[watchers] stopped.")

if __name__ == "__main__":
    main()
