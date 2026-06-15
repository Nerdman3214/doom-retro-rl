import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env import DoomEnv


def main():
    env = DoomEnv(
        launch_doom=False,
        record=False,
    )

    obs, info = env.reset()

    print("[shared_wall_test] reset info:", info)
    print("[shared_wall_test] old_wall_scale:", env.old_wall_reward_scale)
    print("[shared_wall_test] old_stuck_scale:", env.old_stuck_reward_scale)

    for step in range(150):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})

        print(
            f"[shared_wall_test] step={step} "
            f"reward={reward:.4f} "
            f"shared_wall={debug.get('shared_wall_component')} "
            f"shared_wall_total={debug.get('shared_wall_total')} "
            f"after_wall={debug.get('reward_after_shared_wall')} "
            f"old_wall_scale={debug.get('old_wall_reward_scale')} "
            f"old_stuck_scale={debug.get('old_stuck_reward_scale')} "
            f"scene={debug.get('scene_info')} "
            f"stage={debug.get('curriculum_stage')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[shared_wall_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
