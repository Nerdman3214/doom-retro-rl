from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from pynput import keyboard, mouse


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"

KEY_MAP = {
    "move_forward": "w",
    "move_backward": "s",
    "strafe_left": "a",
    "strafe_right": "d",
    "use": "e",
}


def parse_action(action: str) -> set[str]:
    action = (action or "").strip()
    if not action or action == "no_op":
        return set()
    return {part.strip() for part in action.split("+") if part.strip()}


def press_button(kb, ms, btn: str):
    if btn == "mouse_left":
        ms.press(mouse.Button.left)
        return
    key = KEY_MAP.get(btn)
    if key:
        kb.press(key)


def release_button(kb, ms, btn: str):
    if btn == "mouse_left":
        ms.release(mouse.Button.left)
        return
    key = KEY_MAP.get(btn)
    if key:
        kb.release(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--mouse-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = DATA_ROOT / args.run / "actions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[teacher_replay] loaded rows={len(rows)} from {csv_path}")
    print("[teacher_replay] Start Doom Retro at the same spawn/state.")
    print("[teacher_replay] Click/focus Doom Retro.")
    print(f"[teacher_replay] replay begins in {args.countdown:.1f}s...")
    time.sleep(args.countdown)

    kb = keyboard.Controller()
    ms = mouse.Controller()

    held = set()
    dt = 1.0 / max(args.fps, 1.0)

    try:
        for i, row in enumerate(rows):
            start = time.time()

            active = parse_action(row.get("keyboard_action", ""))

            if str(row.get("mouse_left", "0")).strip() in {"1", "1.0", "true", "True"}:
                active.add("mouse_left")

            for btn in list(held):
                if btn not in active:
                    if not args.dry_run:
                        release_button(kb, ms, btn)
                    held.discard(btn)

            for btn in active:
                if btn not in held:
                    if not args.dry_run:
                        press_button(kb, ms, btn)
                    held.add(btn)

            try:
                dx = float(row.get("mouse_dx", 0.0)) * args.mouse_scale
                dy = float(row.get("mouse_dy", 0.0)) * args.mouse_scale
            except ValueError:
                dx, dy = 0.0, 0.0

            if not args.dry_run and (dx != 0.0 or dy != 0.0):
                ms.move(dx, dy)

            if i % 50 == 0:
                print(
                    f"[teacher_replay] frame={i} "
                    f"active={'+'.join(sorted(active)) if active else 'no_op'} "
                    f"mouse=({dx:.1f},{dy:.1f})"
                )

            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        for btn in list(held):
            if not args.dry_run:
                release_button(kb, ms, btn)
        print("[teacher_replay] done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[teacher_replay] stopped")
