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

    if getattr(env.controller, "window_id", None) is None:
        print("[action_prior_test] ERROR: Doom Retro window was not detected.")
        env.close()
        return

    for step in range(100):
        valid_actions = env.get_valid_actions()
        action = valid_actions[step % len(valid_actions)]

        obs, reward, terminated, truncated, info = env.step(action)

        debug = info.get("shared_logic_debug", {})
        prior = info.get("action_prior") or debug.get("action_prior") or {}

        print(
            f"[action_prior_test] step={step} "
            f"reward={reward:.4f} "
            f"prior_action={prior.get('action_name')} "
            f"prior_conf={prior.get('confidence')} "
            f"prior_reward={debug.get('action_prior_reward')} "
            f"error={debug.get('shared_logic_error')}"
        )

        time.sleep(0.05)

        if terminated or truncated:
            print("[action_prior_test] reset")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
