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
    print("[forced_move] start:", info)

    # action index 0 = move_forward
    for step in range(80):
        obs, reward, terminated, truncated, info = env.step(0)
        print(
            f"[forced_move] step={step} "
            f"action={info.get('action_name')} "
            f"buttons={info.get('buttons')} "
            f"last_action={info.get('last_action')} "
            f"time={info.get('episode_time')} "
            f"vars={info.get('game_vars')}"
        )
        time.sleep(0.05)

        if terminated or truncated:
            print("[forced_move] reset")
            obs, info = env.reset()

    # action index 3 = turn_right
    for step in range(40):
        obs, reward, terminated, truncated, info = env.step(3)
        print(
            f"[forced_turn] step={step} "
            f"action={info.get('action_name')} "
            f"last_action={info.get('last_action')} "
            f"time={info.get('episode_time')} "
            f"vars={info.get('game_vars')}"
        )
        time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()
