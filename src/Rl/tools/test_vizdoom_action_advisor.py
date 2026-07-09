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

    print("[advisor_test] observation_space:", env.observation_space)
    print("[advisor_test] reset:", info)

    # Repeatedly ask for move_forward.
    # The advisor should allow it when movement is good,
    # then override to turn when forward is blocked/stuck.
    for step in range(120):
        obs, reward, terminated, truncated, info = env.step(0)

        print(
            f"[advisor_test] step={step} "
            f"requested=move_forward "
            f"actual={info.get('action_name')} "
            f"reason={info.get('action_advice_reason')} "
            f"reward={reward:.4f} "
            f"vars={info.get('game_vars')} "
            f"advisor={info.get('action_advice_debug')}"
        )

        time.sleep(0.04)

        if terminated or truncated:
            print("[advisor_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
