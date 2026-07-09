#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

class Keys:
    def __init__(self, window_name=None):
        self.win = None
        if window_name:
            r = run(["xdotool", "search", "--onlyvisible", "--name", window_name])
            ids = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            if ids:
                self.win = ids[0]
                print(f"[combat] using window id: {self.win}")
                run(["xdotool", "windowactivate", "--sync", self.win])
            else:
                print("[combat] warning: no Doom window found; using focused window")

    def cmd(self, action, key):
        if self.win:
            return ["xdotool", action, "--window", self.win, key]
        return ["xdotool", action, key]

    def keydown(self, key):
        run(self.cmd("keydown", key))

    def keyup(self, key):
        run(self.cmd("keyup", key))

    def keytap(self, key):
        run(self.cmd("key", key))

    def release_all(self):
        self.keyup("Left")
        self.keyup("Right")
        self.keyup("Ctrl")


def load_state(path):
    return json.loads(Path(path).read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    ap.add_argument("--hz", type=float, default=12.0)
    ap.add_argument("--deadzone", type=float, default=7.0)
    ap.add_argument("--shoot-cooldown", type=float, default=0.20)
    ap.add_argument("--invert-turn", action="store_true")
    ap.add_argument("--scan-when-no-enemy", action="store_true")
    ap.add_argument("--scan-key", default="Right", choices=["Left", "Right"])
    ap.add_argument("--max-state-age", type=float, default=1.0)
    ap.add_argument("--window-name", default="DOOM|Doom|doom|Freedoom|freedoom")
    args = ap.parse_args()

    keys = Keys(args.window_name)
    dt = 1.0 / max(args.hz, 1.0)
    last_shot = 0.0
    held = None

    print("[combat] keyboard combat controller running")
    print("[combat] Ctrl+C to stop")
    print("[combat] state:", args.state_json)

    try:
        while True:
            now = time.time()

            try:
                age = now - os.path.getmtime(args.state_json)
                s = load_state(args.state_json)
            except Exception as e:
                print("[combat] waiting for state:", e)
                want = None
                action = "missing_state"
                age = 999

                if held:
                    keys.keyup(held)
                    held = None

                time.sleep(dt)
                continue

            visible = int(s.get("visible_enemy_count") or 0)
            alive_enemies = int(s.get("alive_enemy_count") or 0)
            crosshair = bool(s.get("enemy_in_crosshair"))
            can_fire = bool(s.get("can_autofire"))
            angle_err = float(s.get("nearest_enemy_angle_error") or 0.0)
            dist = s.get("nearest_enemy_distance")
            health = int(s.get("health") or 0)
            kills = int(s.get("kills") or 0)
            bridge = int(s.get("bridge_version") or 0)

            want = None
            action = "none"

            if age > args.max_state_age:
                action = "stale_state"

            elif bridge < 2:
                action = "bridge_v1_only"

            elif health <= 0:
                action = "dead"

            elif crosshair and can_fire:
                action = "shoot"
                if now - last_shot >= args.shoot_cooldown:
                    keys.keytap("Ctrl")
                    last_shot = now

            elif visible > 0:
                if abs(angle_err) <= args.deadzone:
                    action = "centered_wait"
                else:
                    turn_right = angle_err > 0
                    if args.invert_turn:
                        turn_right = not turn_right

                    want = "Right" if turn_right else "Left"
                    action = "turn_" + want.lower()

            elif args.scan_when_no_enemy and alive_enemies > 0:
                want = args.scan_key
                action = "scan_" + want.lower()

            else:
                action = "no_enemy"

            if want != held:
                if held:
                    keys.keyup(held)
                held = want
                if held:
                    keys.keydown(held)

            print(
                f"[combat] action={action} bridge={bridge} age={age:.2f}s "
                f"alive_enemies={alive_enemies} visible={visible} "
                f"crosshair={crosshair} can_fire={can_fire} "
                f"err={angle_err:.1f} dist={dist} health={health} kills={kills}"
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[combat] stopping")
    finally:
        keys.release_all()

if __name__ == "__main__":
    main()
