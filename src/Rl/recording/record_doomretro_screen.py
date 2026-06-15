import argparse
import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "vision_dataset_v2" / "manual_clip_sources"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    if args.name is None:
        args.name = datetime.now().strftime("doomretro_human_%Y%m%d_%H%M%S")

    out_dir = OUT_ROOT / args.name / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_delay = 1.0 / max(args.fps, 1.0)
    max_frames = int(args.duration * args.fps)

    print(f"[doomretro_recorder] saving to: {out_dir}")
    print(f"[doomretro_recorder] fps={args.fps} duration={args.duration}s max_frames={max_frames}")
    print("[doomretro_recorder] Start Doom Retro / focus the game window now.")
    print("[doomretro_recorder] Recording begins in 3 seconds...")
    time.sleep(3)

    with mss.mss() as sct:
        if args.left is not None:
            region = {
                "left": args.left,
                "top": args.top,
                "width": args.width,
                "height": args.height,
            }
        else:
            monitors = sct.monitors
            if args.monitor >= len(monitors):
                raise SystemExit(f"Monitor {args.monitor} not found. Available: 1-{len(monitors)-1}")
            region = monitors[args.monitor]

        print(f"[doomretro_recorder] region={region}")

        start = time.time()

        for i in range(max_frames):
            now = time.time()
            if now - start >= args.duration:
                break

            shot = sct.grab(region)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

            out_path = out_dir / f"frame_{i:06d}.png"
            img.save(out_path)

            if i % 50 == 0:
                print(f"[doomretro_recorder] saved frame {i}")

            elapsed = time.time() - now
            sleep_time = max(0.0, frame_delay - elapsed)
            time.sleep(sleep_time)

    print(f"[doomretro_recorder] done. frames saved to: {out_dir}")


if __name__ == "__main__":
    main()
