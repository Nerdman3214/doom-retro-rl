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

    print("[shared_logic_test] observation_space:", env.observation_space)
    print("[shared_logic_test] reset info:", info)

    # Force movement first.
    for step in range(60):
        obs, reward, terminated, truncated, info = env.step(0)

        print(
            f"[shared_logic_test] step={step} "
            f"action={info.get('action_name')} "
            f"reward={reward:.4f} "
            f"vars={info.get('game_vars')} "
            f"debug={info.get('reward_debug')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            obs, info = env.reset()

    # Then force turning to test wall/depth response.
    for step in range(30):
        obs, reward, terminated, truncated, info = env.step(3)

        print(
            f"[shared_logic_turn] step={step} "
            f"action={info.get('action_name')} "
            f"reward={reward:.4f} "
            f"debug={info.get('reward_debug')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
