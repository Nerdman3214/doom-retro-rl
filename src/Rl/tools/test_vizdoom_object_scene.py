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
        use_labels=True,
        use_automap=False,
        observation_mode="compact",
        frame_skip=4,
        max_episode_steps=1000,
    )

    obs, info = env.reset()

    print("[object_scene_test] observation_space:", env.observation_space)
    print("[object_scene_test] reset:", info)

    for step in range(300):
        # Mix movement, turning, and shooting.
        if step % 40 in [0, 1, 2, 3, 4]:
            action = 3  # turn_right
        elif step % 25 in [0, 1, 2]:
            action = 6  # shoot
        else:
            action = 0  # move_forward

        obs, reward, terminated, truncated, info = env.step(action)

        reward_debug = info.get("reward_debug", {})
        advisor_debug = info.get("action_advice_debug", {})

        object_summary = reward_debug.get("object_summary", {})
        scene_info = reward_debug.get("scene_info", {})

        print(
            f"[object_scene_test] step={step} "
            f"action={info.get('action_name')} "
            f"reason={info.get('action_advice_reason')} "
            f"reward={reward:.4f} "
            f"scene={scene_info} "
            f"objects={object_summary} "
            f"vars={info.get('game_vars')}"
        )

        time.sleep(0.03)

        if terminated or truncated:
            print("[object_scene_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
