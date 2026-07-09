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

    def base(self, action, key):
        if self.win:
            return ["xdotool", action, "--window", self.win, key]
        return ["xdotool", action, key]

    def keydown(self, key):
        run(self.base("keydown", key))

    def keyup(self, key):
        run(self.base("keyup", key))

    def keytap(self, key):
        run(self.base("key", key))

    def pulse(self, key, seconds):
        self.keydown(key)
        time.sleep(seconds)
        self.keyup(key)

    def release_all(self):
        for k in ["Left", "Right", "Ctrl", "Control_L"]:
            self.keyup(k)

def load_state(path):
    return json.loads(Path(path).read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-json", default="/tmp/doomretro_ai_state.json")
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--aim-deadzone", type=float, default=2.0)
    ap.add_argument("--scan-when-no-enemy", action="store_true")
    ap.add_argument("--scan-key", default="Right", choices=["Left", "Right"])
    ap.add_argument("--scan-pulse", type=float, default=0.12)
    ap.add_argument("--turn-pulse", type=float, default=0.08)
    ap.add_argument("--micro-pulse", type=float, default=0.045)
    ap.add_argument("--shoot-cooldown", type=float, default=0.28)
    ap.add_argument("--fire-key", default="Ctrl")
    ap.add_argument("--fire-hold", type=float, default=0.06)
    ap.add_argument("--invert-turn", action="store_true")
    ap.add_argument("--max-state-age", type=float, default=2.0)
    ap.add_argument("--window-name", default="DOOM|Doom|doom|Freedoom|freedoom")
    args = ap.parse_args()

    keys = Keys(args.window_name)
    dt = 1.0 / max(args.hz, 1.0)
    last_shot = 0.0
    last_micro_dir = "Right"

    print("[combat] pulse keyboard combat controller running")
    print("[combat] Ctrl+C to stop")
    print("[combat] state:", args.state_json)

    try:
        while True:
            loop_start = time.time()

            try:
                age = loop_start - os.path.getmtime(args.state_json)
                s = load_state(args.state_json)
            except Exception as e:
                print("[combat] waiting for state:", e)
                keys.release_all()
                time.sleep(dt)
                continue

            bridge = int(s.get("bridge_version") or 0)
            visible = int(s.get("visible_enemy_count") or 0)
            alive_enemies = int(s.get("alive_enemy_count") or 0)
            crosshair = bool(s.get("enemy_in_crosshair"))
            can_fire = bool(s.get("can_autofire"))
            angle_err = float(s.get("nearest_enemy_angle_error") or 0.0)
            dist = s.get("nearest_enemy_distance")
            health = int(s.get("health") or 0)
            kills = int(s.get("kills") or 0)

            action = "none"

            if age > args.max_state_age:
                action = "stale_state"
                keys.release_all()

            elif bridge < 2:
                action = "bridge_v1_only"
                keys.release_all()

            elif health <= 0:
                action = "dead"
                keys.release_all()

            elif crosshair and can_fire:
                now = time.time()
                if now - last_shot >= args.shoot_cooldown:
                    action = "shoot"
                    keys.pulse(args.fire_key, args.fire_hold)
                    last_shot = now
                else:
                    action = "shoot_cooldown"

            elif visible > 0:
                turn_right = angle_err > 0
                if args.invert_turn:
                    turn_right = not turn_right

                if abs(angle_err) > args.aim_deadzone:
                    key = "Right" if turn_right else "Left"
                    action = "aim_" + key.lower()
                    keys.pulse(key, args.turn_pulse)
                    last_micro_dir = key
                else:
                    # Enemy is visible, but not in crosshair. Don't wait forever:
                    # make tiny correction pulses until crosshair becomes true.
                    key = "Right" if angle_err >= 0 else "Left"
                    if args.invert_turn:
                        key = "Left" if key == "Right" else "Right"
                    action = "micro_" + key.lower()
                    keys.pulse(key, args.micro_pulse)
                    last_micro_dir = key

            elif args.scan_when_no_enemy and alive_enemies > 0:
                action = "scan_" + args.scan_key.lower()
                keys.pulse(args.scan_key, args.scan_pulse)

            else:
                action = "no_enemy"
                keys.release_all()

            print(
                f"[combat] action={action} bridge={bridge} age={age:.2f}s "
                f"alive={alive_enemies} visible={visible} crosshair={crosshair} "
                f"can_fire={can_fire} err={angle_err:.1f} dist={dist} "
                f"health={health} kills={kills}"
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\n[combat] stopping")
    finally:
        keys.release_all()

if __name__ == "__main__":
    main()
