import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


def main():
    env = VizDoomEnv(visible=False, max_episode_steps=100)

    obs, info = env.reset()

    print("[test] reset info:", info)
    print("[test] obs keys:", obs.keys())

    for key, value in obs.items():
        print("[test]", key, value.shape, value.dtype, value.min(), value.max())

    total_reward = 0.0

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"[test] step={step} "
            f"action={info.get('action_name')} "
            f"reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated} "
            f"vars={info.get('game_vars')}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("[test] total_reward:", total_reward)


if __name__ == "__main__":
    main()
