"""import_video_demo.py — Import a gameplay video as behaviour-cloning data.

Run from src/Rl/:
    python scripts/import_video_demo.py --video /path/to/gameplay.mp4

What it does:
  1. Opens the video and plays it at ~8 fps in a window.
  2. You label each frame by holding a key that matches the action:
        W  = move_forward          S  = move_backward
        A  = turn_left             D  = turn_right
        F  = shoot                 E  = use (open door)
        1  = move_forward_shoot    2  = turn_left_shoot
        3  = turn_right_shoot      SPACE = skip frame
        Q  = quit / finish
  3. The current label persists until you press a new key — so you
     only need to press once per action-segment, not once per frame.
  4. Saves labelled frames as a .pkl demo compatible with
     train_behavior_clone.py.

Output:
    datasets/human_demos/video_demo_<timestamp>.pkl

Then run:
    python training/train_behavior_clone.py
"""

import os
import pickle
import sys
from datetime import datetime

import cv2
import numpy as np

DEMO_DIR    = "datasets/human_demos"
FRAME_SIZE  = (84, 84)
PLAYBACK_MS = 120           # ms per frame (~8 fps for labelling comfort)

KEY_MAP = {
    ord("w"): "move_forward",
    ord("s"): "move_backward",
    ord("a"): "turn_left",
    ord("d"): "turn_right",
    ord("f"): "shoot",
    ord("e"): "use",
    ord("1"): "move_forward_shoot",
    ord("2"): "turn_left_shoot",
    ord("3"): "turn_right_shoot",
}

ACTION_LABELS = {
    "move_forward":       "W",
    "move_backward":      "S",
    "turn_left":          "A",
    "turn_right":         "D",
    "shoot":              "F",
    "use":                "E",
    "move_forward_shoot": "1",
    "turn_left_shoot":    "2",
    "turn_right_shoot":   "3",
}

HELP = (
    "W=fwd  S=back  A=left  D=right  F=shoot  "
    "E=use  1=fwd+shoot  2=left+shoot  3=right+shoot  "
    "SPACE=skip  Q=done"
)


def _draw_overlay(frame_bgr: np.ndarray, action: str,
                  frame_idx: int, total: int) -> np.ndarray:
    """Return a display-sized copy with HUD overlay."""
    display = cv2.resize(frame_bgr, (420, 420), interpolation=cv2.INTER_NEAREST)
    progress = int(frame_idx / max(total, 1) * 420)
    cv2.rectangle(display, (0, 410), (progress, 420), (0, 200, 0), -1)

    key_hint = ACTION_LABELS.get(action, "?")
    cv2.rectangle(display, (0, 0), (420, 36), (0, 0, 0), -1)
    cv2.putText(display, f"[{key_hint}] {action}   frame {frame_idx}/{total}",
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)

    cv2.rectangle(display, (0, 384), (420, 420), (20, 20, 20), -1)
    cv2.putText(display, HELP, (4, 415),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)
    return display


def label_video(video_path: str) -> list[tuple[np.ndarray, str]]:
    """Play video and return list of (frame_84x84x3, action_str)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  Frames: {total}  — label the frame that is currently showing.")
    print(f"  Controls: {HELP}\n")

    current_action = "move_forward"
    labelled: list[tuple[np.ndarray, str]] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small_frame  = cv2.resize(frame_rgb, FRAME_SIZE,
                                  interpolation=cv2.INTER_AREA)

        display = _draw_overlay(frame_bgr, current_action, frame_idx, total)
        cv2.imshow("Import Video Demo", display)

        key = cv2.waitKey(PLAYBACK_MS) & 0xFF

        if key == ord("q"):
            print(f"\n  Stopped at frame {frame_idx}/{total}.")
            break

        if key == ord(" "):
            # Skip frame — don't record it
            frame_idx += 1
            continue

        if key in KEY_MAP:
            current_action = KEY_MAP[key]

        labelled.append((small_frame, current_action))
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return labelled


def save_demo(labelled: list[tuple[np.ndarray, str]]):
    os.makedirs(DEMO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = os.path.join(DEMO_DIR, f"video_demo_{stamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump(labelled, f)
    print(f"\n  Saved {len(labelled)} labelled frames → {path}")
    return path


def main():
    args = sys.argv[1:]

    if "--video" not in args:
        print(__doc__)
        print("Usage:  python scripts/import_video_demo.py --video <path>")
        return

    idx = args.index("--video")
    if idx + 1 >= len(args):
        print("ERROR: --video requires a path argument.")
        return

    video_path = args[idx + 1]
    if not os.path.exists(video_path):
        print(f"ERROR: File not found: {video_path}")
        return

    print("\n" + "=" * 56)
    print("  DOOM VIDEO DEMO IMPORTER")
    print("=" * 56)
    print("Label what the player is doing in each frame.")
    print("Hold down a key to label a run of frames at once.")
    print("Press Q when you are done (partial import is saved).\n")

    labelled = label_video(video_path)

    if not labelled:
        print("No frames labelled — nothing saved.")
        return

    save_demo(labelled)

    print("\nNow train the agent to copy these moves:")
    print("  python training/train_behavior_clone.py")
    print()

    # Quick summary
    from collections import Counter
    counts = Counter(a for _, a in labelled)
    print("Label breakdown:")
    for action, count in counts.most_common():
        bar = "█" * min(40, count // 2)
        print(f"  {action:25s} {count:5d}  {bar}")


if __name__ == "__main__":
    main()
