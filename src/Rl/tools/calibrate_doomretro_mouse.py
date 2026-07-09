#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

from pynput.mouse import Controller


def read_state(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def wrap180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def wait_angle(path, seconds=0.20):
    time.sleep(seconds)
    s = read_state(path)
    if not s:
        return None
    return float(s.get("angle", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    ap.add_argument("--out", default="configs/doomretro_mouse_calibration.json")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--pixels", type=int, default=20)
    ap.add_argument("--pause", type=float, default=0.25)
    args = ap.parse_args()

    mouse = Controller()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("[calibrate] Focus/click Doom Retro now.")
    print("[calibrate] The tool will move the mouse right, then left.")
    print("[calibrate] Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"[calibrate] {i}...")
        time.sleep(1)

    results = []

    for direction_name, sign in [("right", 1), ("left", -1)]:
        print(f"[calibrate] testing {direction_name}")
        for n in range(args.steps):
            a0 = wait_angle(args.state_json, args.pause)
            if a0 is None:
                print("[calibrate] missing state json")
                continue

            mouse.move(sign * args.pixels, 0)

            a1 = wait_angle(args.state_json, args.pause)
            if a1 is None:
                print("[calibrate] missing state json after move")
                continue

            delta = wrap180(a1 - a0)
            per_pixel = delta / float(sign * args.pixels)

            row = {
                "direction": direction_name,
                "pixels": sign * args.pixels,
                "angle_before": a0,
                "angle_after": a1,
                "delta_degrees": delta,
                "degrees_per_mouse_pixel": per_pixel,
            }
            results.append(row)
            print(
                f"[calibrate] {direction_name} pixels={sign * args.pixels:+d} "
                f"angle {a0:.2f}->{a1:.2f} delta={delta:+.2f} "
                f"deg_per_pixel={per_pixel:+.4f}"
            )

    usable = [r["degrees_per_mouse_pixel"] for r in results if abs(r["delta_degrees"]) > 0.01]

    if usable:
        avg = sum(usable) / len(usable)
    else:
        avg = 0.0

    data = {
        "source": "doomretro_mouse_calibration",
        "state_json": args.state_json,
        "pixels_per_test": args.pixels,
        "average_degrees_per_mouse_pixel": avg,
        "invert_needed_for_route": avg < 0,
        "results": results,
        "notes": [
            "Positive mouse pixels are OS mouse movement to the right.",
            "If average_degrees_per_mouse_pixel is negative, right mouse movement decreases Doom angle.",
            "Use this to convert desired turn degrees into mouse pixels."
        ],
    }

    Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"[calibrate] wrote: {args.out}")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
