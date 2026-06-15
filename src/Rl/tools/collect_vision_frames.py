import sys
import os
import time

# Add project root to Python path.
# This lets tools/ scripts import frame_cache from src/Rl.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import frame_cache


OUTPUT_DIR = os.path.join(ROOT_DIR, "vision_dataset", "raw_frames")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Collecting frames. Press Ctrl+C to stop.")
    print(f"Saving frames to: {OUTPUT_DIR}")

    frame_id = 0

    while True:
        raw, w, h = frame_cache.get_raw()

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(path, frame)

        print(path)

        frame_id += 1
        time.sleep(0.25)


if __name__ == "__main__":
    main()