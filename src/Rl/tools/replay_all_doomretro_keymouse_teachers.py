from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pynput import keyboard


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"
REPLAY_SCRIPT = ROOT / "tools" / "replay_doomretro_keymouse_teacher.py"


def count_rows(run_dir: Path) -> int:
    csv_path = run_dir / "actions.csv"
    if not csv_path.exists():
        return 0
    # subtract header
    return max(0, sum(1 for _ in csv_path.open("r", errors="ignore")) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--mouse-scale", type=float, default=1.0)
    parser.add_argument("--auto-quickload", action="store_true", help="Press F9 before each replay instead of waiting for Enter.")
    parser.add_argument("--reset-wait", type=float, default=2.0)
    parser.add_argument("--start", default=None, help="Example: doomretro_easiest_keymouse_completion_001")
    parser.add_argument("--end", default=None, help="Example: doomretro_easiest_keymouse_completion_061")
    args = parser.parse_args()

    runs = sorted(DATA_ROOT.glob("doomretro_easiest_keymouse_completion_*"))

    if args.start:
        runs = [r for r in runs if r.name >= args.start]
    if args.end:
        runs = [r for r in runs if r.name <= args.end]

    valid = []
    skipped = []

    for run in runs:
        rows = count_rows(run)
        if rows >= args.min_rows:
            valid.append((run, rows))
        else:
            skipped.append((run, rows))

    print(f"[teacher_all] valid runs={len(valid)} skipped={len(skipped)} min_rows={args.min_rows}")
    for run, rows in skipped:
        print(f"[teacher_all] skipped {run.name} rows={rows}")

    print()
    print("[teacher_all] IMPORTANT:")
    print("  Before each replay, reset Doom Retro to the same spawn state.")
    print("  Recommended: quicksave at spawn, then press F9/quickload before continuing.")
    print()

    for idx, (run, rows) in enumerate(valid, start=1):
        print("=" * 80)
        print(f"[teacher_all] next {idx}/{len(valid)}: {run.name} rows={rows}")

        if args.auto_quickload:
            print("[teacher_all] auto quickload: pressing F9")
            kb = keyboard.Controller()
            kb.press(keyboard.Key.f9)
            time.sleep(0.05)
            kb.release(keyboard.Key.f9)
            time.sleep(args.reset_wait)
        else:
            input("[teacher_all] Reset Doom Retro to spawn, click/focus the game, then press ENTER here...")

        cmd = [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--run",
            run.name,
            "--fps",
            str(args.fps),
            "--countdown",
            str(args.countdown),
            "--mouse-scale",
            str(args.mouse_scale),
        ]

        subprocess.run(cmd, cwd=str(ROOT))

    print("[teacher_all] done")


if __name__ == "__main__":
    main()
