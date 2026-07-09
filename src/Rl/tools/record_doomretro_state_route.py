#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def angle_delta(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)

def dist(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    p.add_argument("--out", default="")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--min-dist", type=float, default=6.0)
    p.add_argument("--min-angle", type=float, default=8.0)
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out) if args.out else Path("routes") / f"doomretro_e1m1_route_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[route_record] state:", args.state_json)
    print("[route_record] out:  ", out)
    print("[route_record] Focus/play Doom during countdown.")
    for i in range(args.countdown, 0, -1):
        print(f"[route_record] starting in {i}...")
        time.sleep(1)

    dt = 1.0 / max(args.hz, 1.0)
    start = time.time()
    last = None
    rows = 0

    with out.open("w") as f:
        try:
            while time.time() - start < args.seconds:
                s = read_json(args.state_json)
                now = time.time() - start

                if s and "x" in s and "y" in s:
                    keep = False
                    if last is None:
                        keep = True
                    else:
                        moved = dist(s, last)
                        turned = angle_delta(float(s.get("angle", 0.0)), float(last.get("angle", 0.0)))
                        if moved >= args.min_dist or turned >= args.min_angle:
                            keep = True

                    if keep:
                        row = {
                            "i": rows,
                            "t": round(now, 3),
                            "x": float(s["x"]),
                            "y": float(s["y"]),
                            "angle": float(s.get("angle", 0.0)),
                            "health": int(s.get("health", 0)),
                            "armor": int(s.get("armor", 0)),
                            "ammo_clip": int(s.get("ammo_clip", 0)),
                            "kills": int(s.get("kills", 0)),
                            "alive": bool(s.get("alive", False)),
                        }
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        last = dict(s)
                        rows += 1

                time.sleep(dt)
        except KeyboardInterrupt:
            print()
            print("[route_record] Ctrl+C received.")

    print("[route_record] rows:", rows)
    print("[route_record] wrote:", out)

if __name__ == "__main__":
    main()
