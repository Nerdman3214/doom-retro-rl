"""rate_clips.py — Rate individual clips as good / ok / bad.

Run from src/Rl/:
    python scripts/rate_clips.py              # rate only unrated clips
    python scripts/rate_clips.py --re-rate    # re-rate everything

How it works:
  1. Shows each clip one at a time in a window.
  2. You rate it: good / ok / bad / skip.
  3. Good vs bad pairs are written to human_preferences.pkl so that
     train_preference_model.py can learn from your ratings immediately.
"""

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preview_clip import play_clip

CLIPS_DIR    = "clips"
RATINGS_FILE = "clip_ratings.pkl"
PREFS_FILE   = "human_preferences.pkl"

SCORE = {"good": 1.0, "ok": 0.5, "bad": 0.0}
SCORE_LABEL = {1.0: "good", 0.5: "ok", 0.0: "bad"}


# ── helpers ────────────────────────────────────────────────────────────────

def load_ratings() -> dict:
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_ratings(ratings: dict):
    with open(RATINGS_FILE, "wb") as f:
        pickle.dump(ratings, f)
    print(f"  Saved {len(ratings)} ratings → {RATINGS_FILE}")


def ratings_to_prefs(ratings: dict) -> list:
    """Create preference pairs from every good-clip vs every bad-clip pair."""
    good = [n for n, s in ratings.items() if s >= 1.0]
    bad  = [n for n, s in ratings.items() if s <= 0.0]

    pairs = []
    for g_name in good:
        g_path = os.path.join(CLIPS_DIR, g_name)
        if not os.path.exists(g_path):
            continue
        with open(g_path, "rb") as f:
            c1 = pickle.load(f)

        for b_name in bad:
            b_path = os.path.join(CLIPS_DIR, b_name)
            if not os.path.exists(b_path):
                continue
            with open(b_path, "rb") as f:
                c2 = pickle.load(f)

            # Sub-sample to keep array size manageable
            f1 = np.array([frame for frame, _ in c1[::3]])
            f2 = np.array([frame for frame, _ in c2[::3]])

            pairs.append({
                "clip1_frames": f1,
                "clip2_frames": f2,
                "label": 1.0,   # clip1 (good) preferred over clip2 (bad)
            })

    return pairs


def _update_prefs(ratings: dict):
    new_pairs = ratings_to_prefs(ratings)
    existing = []
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "rb") as f:
            existing = pickle.load(f)

    combined = existing + new_pairs
    with open(PREFS_FILE, "wb") as f:
        pickle.dump(combined, f)

    print(f"  Added {len(new_pairs)} new preference pairs → {PREFS_FILE} "
          f"(total: {len(combined)})")
    print("  Run:  python training/train_preference_model.py  to apply them.")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    re_rate = "--re-rate" in sys.argv

    clip_files = sorted(glob.glob(os.path.join(CLIPS_DIR, "clip_*.pkl")))
    if not clip_files:
        print("No clips found in clips/.  Run train_rl_agent.py first.")
        return

    ratings = load_ratings()
    print(f"\nFound {len(clip_files)} clips, {len(ratings)} already rated.")
    print("Controls:  good / ok / bad / skip / quit\n")

    changed = False

    for clip_path in clip_files:
        name = os.path.basename(clip_path)

        if name in ratings and not re_rate:
            label = SCORE_LABEL.get(ratings[name], "?")
            ans = input(f"  {name} already rated [{label}] — re-rate? (y/N): "
                        ).strip().lower()
            if ans != "y":
                continue

        print(f"\n--- Playing: {name} ---")
        play_clip(clip_path)

        while True:
            choice = input("Rate (good / ok / bad / skip / quit): ").strip().lower()

            if choice == "quit":
                save_ratings(ratings)
                if changed:
                    _update_prefs(ratings)
                return

            if choice == "skip":
                break

            if choice in SCORE:
                ratings[name] = SCORE[choice]
                changed = True
                print(f"  ✓ Rated '{choice}'.")
                break

            print("  Please enter: good, ok, bad, skip, or quit.")

    save_ratings(ratings)
    if changed:
        _update_prefs(ratings)
    else:
        print("No new ratings — preference file unchanged.")


if __name__ == "__main__":
    main()
