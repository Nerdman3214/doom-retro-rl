"""Human preference labeling tool.

Displays pairs of trajectory clips side-by-side.
Press '1' if left clip is better, '2' if right clip is better, 's' to skip.
Results saved to human_preferences.pkl.
"""
import sys
import os
import glob
import random
import pickle
import numpy as np
import cv2

TRAJ_DIR = "trajectories"
CLIP_LEN = 16
OUTPUT_FILE = "human_preferences.pkl"
N_COMPARISONS = 100


def load_trajectories():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "traj_*.npz")))
    trajs = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        frames = data["frames"]
        if len(frames) >= CLIP_LEN:
            trajs.append(frames)
    return trajs


def sample_clip(traj):
    start = random.randint(0, len(traj) - CLIP_LEN)
    return traj[start:start + CLIP_LEN]


def show_pair(clip1, clip2):
    """Play two clips side-by-side and get human preference."""
    print("\nShowing clip pair. Press '1' (left better), "
          "'2' (right better), 's' (skip), 'q' (quit)")

    for i in range(CLIP_LEN):
        f1 = clip1[i]
        f2 = clip2[i]

        # Convert RGB to BGR for OpenCV
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        f2_bgr = cv2.cvtColor(f2, cv2.COLOR_RGB2BGR)

        # Scale up for visibility
        f1_big = cv2.resize(f1_bgr, (336, 336),
                            interpolation=cv2.INTER_NEAREST)
        f2_big = cv2.resize(f2_bgr, (336, 336),
                            interpolation=cv2.INTER_NEAREST)

        # Add labels
        cv2.putText(f1_big, "1 (Left)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2_big, "2 (Right)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack([f1_big, f2_big])
        cv2.imshow("Preference Labeling", combined)

        key = cv2.waitKey(150) & 0xFF
        if key == ord('1'):
            cv2.destroyAllWindows()
            return 1.0
        elif key == ord('2'):
            cv2.destroyAllWindows()
            return 0.0
        elif key == ord('s'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"

    # After full playback, wait for input
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'):
            cv2.destroyAllWindows()
            return 1.0
        elif key == ord('2'):
            cv2.destroyAllWindows()
            return 0.0
        elif key == ord('s'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"


def main():
    trajs = load_trajectories()
    if len(trajs) < 2:
        print(f"Need at least 2 trajectories in {TRAJ_DIR}/. "
              "Run train_rl_agent.py first.")
        return

    print(f"Loaded {len(trajs)} trajectories.")
    print(f"Will present up to {N_COMPARISONS} pairs for labeling.")
    print("Controls: '1' = left better, '2' = right better, "
          "'s' = skip, 'q' = quit\n")

    # Load existing preferences if any
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            preferences = pickle.load(f)
        print(f"Resuming — {len(preferences)} labels already saved.")
    else:
        preferences = []

    labeled = 0
    for i in range(N_COMPARISONS):
        t1 = random.choice(trajs)
        t2 = random.choice(trajs)
        clip1 = sample_clip(t1)
        clip2 = sample_clip(t2)

        print(f"[{i + 1}/{N_COMPARISONS}] "
              f"(labeled so far: {len(preferences)})")

        result = show_pair(clip1, clip2)

        if result == "quit":
            break
        elif result is None:
            continue
        else:
            preferences.append({
                "clip1_frames": clip1,
                "clip2_frames": clip2,
                "label": result
            })
            labeled += 1

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(preferences, f)
    print(f"\nDone. Labeled {labeled} new pairs. "
          f"Total: {len(preferences)} in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
