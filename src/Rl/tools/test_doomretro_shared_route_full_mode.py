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

    # Route/curriculum ON.
    env.use_shared_route_curriculum_reward = True

    # Full shared replacement OFF for this first test.
    env.use_full_shared_reward_mode = False

    obs, info = env.reset()

    if getattr(env.controller, "window_id", None) is None:
        print("[route_full_test] ERROR: Doom Retro window was not detected.")
        print("[route_full_test] Start Doom Retro first, then rerun this test.")
        env.close()
        return

    print("[route_full_test] reset info:", info)
    print("[route_full_test] full_shared_mode:", env.use_full_shared_reward_mode)

    for step in range(180):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})

        print(
            f"[route_full_test] step={step} "
            f"reward={reward:.4f} "
            f"route={debug.get('shared_route_component')} "
            f"route_total={debug.get('shared_route_curriculum_total')} "
            f"after_route={debug.get('reward_after_shared_route_curriculum')} "
            f"stage={debug.get('shared_curriculum_stage')} "
            f"weights={debug.get('shared_curriculum_weights')} "
            f"full_mode={debug.get('use_full_shared_reward_mode')} "
            f"full_raw={debug.get('full_shared_reward_raw')} "
            f"final={debug.get('final_reward_output')} "
            f"old_route_scale={debug.get('old_route_reward_scale')} "
            f"old_curr_scale={debug.get('old_curriculum_reward_scale')} "
            f"scene={debug.get('scene_info')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[route_full_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
