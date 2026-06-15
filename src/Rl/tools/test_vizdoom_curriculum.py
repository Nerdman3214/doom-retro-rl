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
        max_episode_steps=1000,
    )

    obs, info = env.reset()

    print("[curriculum_test] observation_space:", env.observation_space)
    print("[curriculum_test] reset:", info)

    for step in range(400):
        # Mostly move forward, with occasional turns so it does not just hit one wall forever.
        if step % 35 in [0, 1, 2, 3, 4]:
            action = 3  # turn_right
        else:
            action = 0  # move_forward

        obs, reward, terminated, truncated, info = env.step(action)
        debug = info.get("reward_debug", {})

        print(
            f"[curriculum_test] step={step} "
            f"stage={debug.get('curriculum_stage')} "
            f"action={info.get('action_name')} "
            f"reason={info.get('action_advice_reason')} "
            f"reward={reward:.4f} "
            f"route={debug.get('route_zone')} "
            f"tiles={debug.get('visited_tiles')} "
            f"dist={debug.get('distance_moved'):.2f} "
            f"weights={debug.get('curriculum_weights')}"
        )

        time.sleep(0.03)

        if terminated or truncated:
            print("[curriculum_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
