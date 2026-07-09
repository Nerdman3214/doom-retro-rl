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

    print("[shared_movement_test] reset info:", info)

    for step in range(150):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})

        print(
            f"[shared_movement_test] step={step} "
            f"reward={reward:.4f} "
            f"move={debug.get('shared_movement_component')} "
            f"explore={debug.get('shared_exploration_component')} "
            f"move_total={debug.get('shared_movement_exploration_total')} "
            f"after_move={debug.get('reward_after_shared_movement_exploration')} "
            f"blend={debug.get('shared_reward_clipped')} "
            f"final={debug.get('final_reward_after_shared_blend')} "
            f"stage={debug.get('curriculum_stage')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[shared_movement_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
