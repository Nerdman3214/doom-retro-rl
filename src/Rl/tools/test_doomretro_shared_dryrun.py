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

    print("[doomretro_shared_dryrun] reset info:", info)

    for step in range(200):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        shared_debug = info.get("shared_logic_debug", {})

        print(
            f"[doomretro_shared_dryrun] step={step} "
            f"reward={reward:.4f} "
            f"shared_preview={shared_debug.get('shared_reward_preview')} "
            f"scene={shared_debug.get('scene_info')} "
            f"objects={shared_debug.get('object_summary')} "
            f"stage={shared_debug.get('curriculum_stage')} "
            f"error={shared_debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[doomretro_shared_dryrun] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
