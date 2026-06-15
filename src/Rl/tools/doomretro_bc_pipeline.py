from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOOM_BINARY = Path("/home/steven/Downloads/doomretro-master/build/doomretro")
IWAD = Path("/usr/share/games/doom/freedoom1.wad")

RECORDER = ROOT / "recording" / "record_doomretro_screen_keys_mouse.py"
TRAINER = ROOT / "training" / "train_doomretro_keymouse_sequence_imitation.py"
STUDENT = ROOT / "tools" / "run_doomretro_keymouse_sequence_agent.py"


def run(cmd: list[str], *, wait: bool = True):
    print()
    print("[pipeline] running:")
    print(" ".join(cmd))
    print()

    if wait:
        return subprocess.run(cmd, cwd=str(ROOT))
    return subprocess.Popen(cmd, cwd=str(ROOT))


def launch_doom(skill: int):
    if not DOOM_BINARY.exists():
        raise FileNotFoundError(DOOM_BINARY)
    if not IWAD.exists():
        raise FileNotFoundError(IWAD)

    cmd = [
        str(DOOM_BINARY),
        "-iwad",
        str(IWAD),
        "-skill",
        str(skill),
        "-warp",
        "1",
        "1",
    ]

    print("[pipeline] launching Doom Retro...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    time.sleep(5)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=int, default=1, help="1=easiest / please don't hurt me style")
    parser.add_argument("--record-seconds", type=int, default=45)
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--skip-record", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--student-print-every", type=int, default=10)
    args = parser.parse_args()

    doom_proc = None

    try:
        if not args.skip_record:
            doom_proc = launch_doom(args.skill)

            print("[pipeline] Teacher phase.")
            print("[pipeline] Click Doom Retro and play the level.")
            print("[pipeline] Recorder will stop after duration or Ctrl+C.")

            run([
                sys.executable,
                str(RECORDER),
                "--duration",
                str(args.record_seconds),
                "--fps",
                str(args.record_fps),
                "--name",
                time.strftime("doomretro_easiest_keymouse_completion_%Y%m%d_%H%M%S"),
            ])

            print("[pipeline] Teacher recording finished.")

            if doom_proc is not None:
                print("[pipeline] closing Doom Retro teacher window...")
                doom_proc.terminate()
                try:
                    doom_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    doom_proc.kill()
                doom_proc = None

        if not args.skip_train:
            print("[pipeline] Training behavior clone...")
            run([sys.executable, str(TRAINER)])

        if not args.skip_student:
            print("[pipeline] Launching Doom Retro for student...")
            doom_proc = launch_doom(args.skill)

            print("[pipeline] Student phase.")
            print("[pipeline] Click Doom Retro, then press F8 in the student controller.")
            run([
                sys.executable,
                str(STUDENT),
                "--print-every",
                str(args.student_print_every),
            ])

    finally:
        if doom_proc is not None:
            print("[pipeline] cleaning up Doom Retro...")
            doom_proc.terminate()
            try:
                doom_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                doom_proc.kill()


if __name__ == "__main__":
    main()
