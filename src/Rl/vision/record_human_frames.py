from pathlib import Path
import time
import cv2

from observation.observation_builder import ObservationBuilder 


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "vision_dataset" / "human_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    observer = ObservationBuilder()

    print("Recording human gameplay frames.")
    print("Focus DOOM and play normally.")
    print("Press Ctrl+C in this terminal to stop recording.")

    frame_id = 0

    try:
        while True:
            frame = observer.get_frame()

            if frame is not None and frame.size > 0:
                path = OUT_DIR / f"human_frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(path), frame)
                frame_id += 1

                if frame_id % 50 == 0:
                    print(f"Saved {frame_id} frames to {OUT_DIR}")

            time.sleep(0.15)

    except KeyboardInterrupt:
        print(f"\nStopped. Saved {frame_id} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()