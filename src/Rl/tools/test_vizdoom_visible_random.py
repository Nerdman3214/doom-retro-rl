import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


def main():
    env = VizDoomEnv(
        visible=True,
        use_depth=True,
        use_labels=False,
        use_automap=False,
        observation_mode="compact",
        frame_skip=4,
        max_episode_steps=500,
    )

    obs, info = env.reset()

    print("[visible_test] observation_space:", env.observation_space)
    print("[visible_test] obs shape:", obs.shape)
    print("[visible_test] starting random actions...")

    for step in range(300):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"[visible_test] step={step} "
            f"action={info.get('action_name')} "
            f"reward={reward:.4f} "
            f"vars={info.get('game_vars')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[visible_test] episode reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()