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
    return math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])) % 360.0

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
    args = p.parse_args()

    route = load_route(args.route)
    if not route:
        raise SystemExit("[error] route is empty")

    print("[route_watch] route:", args.route)
    print("[route_watch] points:", len(route))
    print("[route_watch] state:", args.state_json)

    dt = 1.0 / max(args.hz, 1.0)

    while True:
        s = read_state(args.state_json)
        if not s or "x" not in s or "y" not in s:
            print("[route_watch] waiting for state...")
            time.sleep(dt)
            continue

        i, d = nearest_index(route, s)
        target_i = min(i + args.lookahead, len(route) - 1)
        target = route[target_i]

        target_heading = heading_to(s, target)
        current_angle = float(s.get("angle", 0.0))
        err = wrap180(target_heading - current_angle)

        status = "OFF_ROUTE" if d > args.offroute else "on_route"
        turn = "left" if err > 8 else "right" if err < -8 else "straight"

        print(
            f"[route_watch] {status} "
            f"wp={i}/{len(route)-1} target={target_i} "
            f"dist={d:6.1f} x={float(s['x']):7.1f} y={float(s['y']):7.1f} "
            f"angle={current_angle:6.1f} target={target_heading:6.1f} "
            f"err={err:7.1f} advice={turn}",
            flush=True,
        )
        time.sleep(dt)

if __name__ == "__main__":
    main()
