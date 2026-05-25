import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


def main():
    env = VizDoomEnv(
        visible=False,
        max_episode_steps=100,
        use_depth=True,
        use_labels=False,
        use_automap=False,
        observation_mode="compact",
    )

    obs, info = env.reset()

    print("[test_native] obs shape:", obs.shape)
    print("[test_native] obs dtype:", obs.dtype)
    print("[test_native] obs min/max:", obs.min(), obs.max())
    print("[test_native] info:", info)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"[test_native] step={step} "
            f"action={info.get('action_name')} "
            f"reward={reward:.4f} "
            f"obs_shape={obs.shape}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()