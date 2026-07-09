#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def key_pulse(key, hold):
    run(["xdotool", "keydown", key])
    time.sleep(hold)
    run(["xdotool", "keyup", key])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--fire-key", default="Ctrl")
    ap.add_argument("--fire-hold", type=float, default=0.10)
    ap.add_argument("--shoot-cooldown", type=float, default=0.45)
    ap.add_argument("--max-state-age", type=float, default=2.0)
    args = ap.parse_args()

    last_shot = 0.0
    dt = 1.0 / max(args.hz, 1.0)

    print("[fire_assist] running")
    print("[fire_assist] Ctrl+C to stop")
    print("[fire_assist] state:", args.state_json)

    try:
        while True:
            now = time.time()

            try:
                age = now - os.path.getmtime(args.state_json)
                s = json.loads(Path(args.state_json).read_text())
            except Exception as e:
                print("[fire_assist] waiting_for_state", e)
                time.sleep(dt)
                continue

            bridge = int(s.get("bridge_version") or 0)
            visible = int(s.get("visible_enemy_count") or 0)
            crosshair = bool(s.get("enemy_in_crosshair"))
            can_fire = bool(s.get("can_autofire"))
            err = float(s.get("nearest_enemy_angle_error") or 0.0)
            dist = s.get("nearest_enemy_distance")
            alive = int(s.get("alive_enemy_count") or 0)
            kills = int(s.get("kills") or 0)
            ammo = int(s.get("ammo_clip") or 0)
            health = int(s.get("health") or 0)

            action = "hold"

            if age > args.max_state_age:
                action = "stale_state"

            elif bridge < 2:
                action = "bridge_v1_only"

            elif health <= 0:
                action = "dead"

            elif crosshair and can_fire and now - last_shot >= args.shoot_cooldown:
                action = "shoot"
                key_pulse(args.fire_key, args.fire_hold)
                last_shot = time.time()

            elif visible == 0:
                action = "no_enemy"

            elif visible > 0 and not crosshair:
                action = "enemy_visible_not_centered"

            else:
                action = "cooldown_or_not_ready"

            print(
                f"[fire_assist] action={action} bridge={bridge} age={age:.2f}s "
                f"alive={alive} visible={visible} crosshair={crosshair} "
                f"can_fire={can_fire} err={err:.1f} dist={dist} "
                f"health={health} ammo={ammo} kills={kills}"
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[fire_assist] stopping")
    finally:
        run(["xdotool", "keyup", args.fire_key])
        run(["xdotool", "keyup", "Ctrl"])
        run(["xdotool", "keyup", "Control_L"])

if __name__ == "__main__":
    main()
