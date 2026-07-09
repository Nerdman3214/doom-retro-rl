import pickle
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from scripts.preview_clip import play_clip


with open("clip_pairs.pkl", "rb") as f:
    pairs = pickle.load(f)


preferences = []


for clip_a, clip_b in pairs:

    print("\nShowing Clip A")
    play_clip(f"clips/{clip_a}")

    print("\nShowing Clip B")
    play_clip(f"clips/{clip_b}")

    choice = input(
        "Which clip is better? (A/B/skip): "
    ).lower()


    if choice in ["a", "b"]:

        preferences.append(
            (clip_a, clip_b, choice)
        )


with open("human_preferences.pkl", "wb") as f:

    pickle.dump(preferences, f)


print("\nPreferences saved successfully.")