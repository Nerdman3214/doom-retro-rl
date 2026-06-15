from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OLD_RECORDER_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"
CLEAN_ROOT = ROOT / "data" / "supervised_doomretro_v1"
RAW_RUNS = CLEAN_ROOT / "raw_runs"

RECORDER = ROOT / "recording" / "record_doomretro_screen_keys_mouse.py"


def next_run_name(kind: str) -> str:
    RAW_RUNS.mkdir(parents=True, exist_ok=True)

    nums = []
    for parent in (RAW_RUNS, OLD_RECORDER_ROOT):
        if not parent.exists():
            continue

        for p in parent.iterdir():
            if not p.is_dir():
                continue

            parts = p.name.split("_")
            if len(parts) >= 2 and parts[0] == "run" and parts[1].isdigit():
                nums.append(int(parts[1]))

    n = max(nums, default=0) + 1
    return f"run_{n:03d}_{kind}"


def safe_move_run(old_path: Path, clean_path: Path, keep_old_copy: bool = False) -> None:
    if not old_path.exists():
        print(f"[auto_record] no temp run to move: {old_path}")
        return

    clean_path.parent.mkdir(parents=True, exist_ok=True)

    if clean_path.exists():
        raise SystemExit(f"[auto_record] clean path already exists: {clean_path}")

    if keep_old_copy:
        shutil.copytree(old_path, clean_path)
        print(f"[auto_record] copied to clean dataset: {clean_path}")
    else:
        shutil.move(str(old_path), str(clean_path))
        print(f"[auto_record] moved to clean dataset: {clean_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        default="complete",
        choices=["complete", "combat", "door_use", "stuck_recovery", "secret", "correction"],
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--name", default=None)
    parser.add_argument("--keep-old-copy", action="store_true")
    args = parser.parse_args()

    OLD_RECORDER_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_RUNS.mkdir(parents=True, exist_ok=True)

    run_name = args.name or next_run_name(args.kind)
    old_path = OLD_RECORDER_ROOT / run_name
    clean_path = RAW_RUNS / run_name

    if old_path.exists():
        raise SystemExit(f"Old recorder path already exists: {old_path}")

    if clean_path.exists():
        raise SystemExit(f"Clean run path already exists: {clean_path}")

    print("=" * 80)
    print(f"[auto_record] run_name: {run_name}")
    print(f"[auto_record] fps: {args.fps}")
    print(f"[auto_record] temporary path: {old_path}")
    print(f"[auto_record] final clean path: {clean_path}")
    print("=" * 80)
    print("[auto_record] Start Doom Retro, click the game, play the route.")
    print("[auto_record] Press Ctrl+C when the run is finished.")
    print("[auto_record] If Ctrl+C appears to stop this wrapper too, it will still move the saved run.")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(RECORDER),
        "--name",
        run_name,
        "--fps",
        str(args.fps),
    ]

    try:
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[auto_record] Ctrl+C reached wrapper. Attempting to save/move completed run...")

    safe_move_run(old_path, clean_path, keep_old_copy=args.keep_old_copy)

    actions_csv = clean_path / "actions.csv"
    frames_dir = clean_path / "frames"

    print("=" * 80)
    print("[auto_record] DONE")
    print(f"[auto_record] run:     {run_name}")
    print(f"[auto_record] actions: {actions_csv}")
    print(f"[auto_record] frames:  {frames_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
