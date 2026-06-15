import pickle
import random


def load_traj(path):

    with open(path, "rb") as f:

        return pickle.load(f)


traj_a = load_traj("traj_a.pkl")
traj_b = load_traj("traj_b.pkl")


print("Which trajectory is better?")
print("Press A or B")

choice = input().lower()

with open("human_preferences.pkl", "ab") as f:

    pickle.dump(
        (traj_a, traj_b, choice),
        f
    )


print("Preference saved.")