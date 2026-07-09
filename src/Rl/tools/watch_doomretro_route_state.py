#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

def load_route(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def read_state(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def wrap180(a):
    return ((a + 180.0) % 360.0) - 180.0

def heading_to(a, b):
    dx = float(b["x"]) - float(a["x"])
    dy = float(b["y"]) - float(a["y"])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(a.get("angle", 0.0))
    return math.degrees(math.atan2(dy, dx)) % 360.0

def distance_xy(a, b):
    return math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))

def nearest_index(route, s):
    best_i = 0
    best_d = 1e9
    x = float(s["x"])
    y = float(s["y"])
    for i, p in enumerate(route):
        d = math.hypot(float(p["x"]) - x, float(p["y"]) - y)
        if d < best_d:
            best_i = i
            best_d = d
    return best_i, best_d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", required=True)
    p.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--lookahead", type=int, default=6)
    p.add_argument("--offroute", type=float, default=80.0)
    p.add_argument("--success-distance", type=float, default=18.0)
    p.add_argument("--success-hold", type=float, default=1.0)
    p.add_argument("--stop-on-success", action="store_true")
    args = p.parse_args()

    route = load_route(args.route)
    if not route:
        raise SystemExit("[error] route is empty")

    final_wp = route[-1]
    final_enter_time = None
    success_reported = False

    print("[route_watch] route:", args.route)
    print("[route_watch] points:", len(route))
    print("[route_watch] state:", args.state_json)
    print("[route_watch] final waypoint:", {"x": final_wp["x"], "y": final_wp["y"], "angle": final_wp.get("angle")})
    print("[route_watch] success rule: final waypoint distance <=", args.success_distance, "for", args.success_hold, "sec")

    dt = 1.0 / max(args.hz, 1.0)

    try:
        while True:
            s = read_state(args.state_json)
            if not s or "x" not in s or "y" not in s:
                print("[route_watch] waiting for state...")
                time.sleep(dt)
                continue

            i, d = nearest_index(route, s)
            final_d = distance_xy(s, final_wp)
            now = time.time()

            in_final_zone = final_d <= args.success_distance or i >= len(route) - 2

            if in_final_zone:
                if final_enter_time is None:
                    final_enter_time = now
                final_hold = now - final_enter_time
            else:
                final_enter_time = None
                final_hold = 0.0

            if in_final_zone and final_hold >= args.success_hold:
                status = "ROUTE_SUCCESS"
                success_reported = True
                print(
                    f"[route_watch] {status} "
                    f"wp={i}/{len(route)-1} final_dist={final_d:5.1f} "
                    f"x={float(s['x']):7.1f} y={float(s['y']):7.1f} "
                    f"angle={float(s.get('angle', 0.0)):6.1f} "
                    f"health={int(s.get('health', 0))} ammo={int(s.get('ammo_clip', 0))}",
                    flush=True,
                )
                if args.stop_on_success:
                    break
                time.sleep(dt)
                continue

            target_i = min(i + args.lookahead, len(route) - 1)
            target = route[target_i]

            target_heading = heading_to(s, target)
            current_angle = float(s.get("angle", 0.0))
            err = wrap180(target_heading - current_angle)

            if in_final_zone:
                status = "FINAL_ZONE"
                turn = "finish"
            else:
                status = "OFF_ROUTE" if d > args.offroute else "on_route"
                turn = "left" if err > 8 else "right" if err < -8 else "straight"

            print(
                f"[route_watch] {status} "
                f"wp={i}/{len(route)-1} target={target_i} "
                f"dist={d:6.1f} final_dist={final_d:5.1f} "
                f"x={float(s['x']):7.1f} y={float(s['y']):7.1f} "
                f"angle={current_angle:6.1f} target={target_heading:6.1f} "
                f"err={err:7.1f} advice={turn}",
                flush=True,
            )
            time.sleep(dt)

    except KeyboardInterrupt:
        print()
        print("[route_watch] Ctrl+C received.")
        if success_reported:
            print("[route_watch] last status: ROUTE_SUCCESS was reached.")
        else:
            print("[route_watch] last status: stopped before route success.")

if __name__ == "__main__":
    main()
