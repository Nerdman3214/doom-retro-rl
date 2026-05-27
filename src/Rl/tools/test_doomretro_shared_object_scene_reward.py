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

    print("[shared_object_scene_test] reset info:", info)
    print("[shared_object_scene_test] old_object_scale:", env.old_object_reward_scale)
    print("[shared_object_scene_test] old_scene_scale:", env.old_scene_reward_scale)
    print("[shared_object_scene_test] old_combat_scale:", env.old_combat_reward_scale)

    for step in range(150):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})

        print(
            f"[shared_object_scene_test] step={step} "
            f"reward={reward:.4f} "
            f"shared_obj_scene={debug.get('shared_object_scene_component')} "
            f"shared_combat={debug.get('shared_combat_component')} "
            f"shared_use={debug.get('shared_use_component')} "
            f"shared_total={debug.get('shared_object_scene_total')} "
            f"after_obj_scene={debug.get('reward_after_shared_object_scene')} "
            f"objects={debug.get('object_summary')} "
            f"scene={debug.get('scene_info')} "
            f"old_obj_scale={debug.get('old_object_reward_scale')} "
            f"old_scene_scale={debug.get('old_scene_reward_scale')} "
            f"old_combat_scale={debug.get('old_combat_reward_scale')} "
            f"stage={debug.get('curriculum_stage')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[shared_object_scene_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
