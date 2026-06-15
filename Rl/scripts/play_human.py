import sys
import os
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from stable_baselines3 import PPO
from env.doom_env import DoomEnv

CHECKPOINT = os.path.join(dirname(dirname(abspath(__file__))), "checkpoints", "doom_rl_agent")

if __name__ == "__main__":
    env = DoomEnv()

    if os.path.exists(CHECKPOINT + ".zip"):
        print(f"Loading trained agent from {CHECKPOINT}.zip ...")
        model = PPO.load(CHECKPOINT, env=env)
    else:
        print("No trained checkpoint found. Run training/train_rl_agent.py first.")
        print("Running with random actions for now.")
        model = None

    obs, _ = env.reset()
    print("AI is playing DOOM. Press Ctrl+C to stop.")
    try:
        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        env.close()
