import sys
import time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


def main():
    env = VizDoomEnv(
        visible=True,
        use_depth=True,
        use_labels=True,
        use_automap=False,
        observation_mode="compact",
        frame_skip=4,
        max_episode_steps=1000,
    )

    obs, info = env.reset()

    names = Counter()
    categories = Counter()

    for step in range(400):
        if step % 40 < 8:
            action = 3  # turn_right
        elif step % 70 < 6:
            action = 6  # shoot
        else:
            action = 0  # move_forward

        obs, reward, terminated, truncated, info = env.step(action)

        reward_debug = info.get("reward_debug", {})
        obj_summary = reward_debug.get("object_summary", {})

        for name in obj_summary.get("visible_names", []):
            names[name] += 1

        for category in obj_summary.get("categories", []):
            categories[category] += 1

        if step % 25 == 0:
            print(f"\n[inspect_objects] step={step}")
            print("scene:", reward_debug.get("scene_info"))
            print("object_summary:", obj_summary)
            print("top names:", names.most_common(20))
            print("top categories:", categories.most_common(20))

        time.sleep(0.03)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    print("\n[inspect_objects] FINAL top names:")
    for name, count in names.most_common(50):
        print(f"{name}: {count}")

    print("\n[inspect_objects] FINAL top categories:")
    for category, count in categories.most_common(20):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
