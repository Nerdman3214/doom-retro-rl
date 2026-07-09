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

    env.use_shared_route_curriculum_reward = True
    env.use_full_shared_reward_mode = True

    # Keep full mode conservative for the first test.
    env.full_shared_reward_scale = 1.0
    env.full_shared_reward_clip = 3.0

    obs, info = env.reset()

    if getattr(env.controller, "window_id", None) is None:
        print("[full_shared_short_test] ERROR: Doom Retro window was not detected.")
        env.close()
        return

    print("[full_shared_short_test] full_shared_mode:", env.use_full_shared_reward_mode)
    print("[full_shared_short_test] full_shared_clip:", env.full_shared_reward_clip)

    for step in range(80):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})

        print(
            f"[full_shared_short_test] step={step} "
            f"reward={reward:.4f} "
            f"shared_preview={debug.get('shared_reward_preview')} "
            f"full_raw={debug.get('full_shared_reward_raw')} "
            f"full_scaled={debug.get('full_shared_reward_scaled')} "
            f"full_clipped={debug.get('full_shared_reward_clipped')} "
            f"base_replaced={debug.get('base_reward_replaced_by_full_shared')} "
            f"final={debug.get('final_reward_output')} "
            f"route={debug.get('shared_route_component')} "
            f"stage={debug.get('shared_curriculum_stage') or debug.get('curriculum_stage')} "
            f"scene={debug.get('scene_info')} "
            f"objects={debug.get('object_summary')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[full_shared_short_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
