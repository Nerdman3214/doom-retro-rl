from stable_baselines3 import PPO
from env.doom_env import DoomEnv
from wrappers.normalize_wrapper import NormalizeObservationWrapper

MODEL_PATH = "checkpoints/doom_rl_agent.zip"

env = DoomEnv(launch_doom=True, record=False)
env = NormalizeObservationWrapper(env)

model = PPO.load(MODEL_PATH, env=env)

episodes = 5
scores = []

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    scores.append(total_reward)
    print(f"Episode {ep + 1}: reward={total_reward:.2f}")

print(f"Average reward: {sum(scores) / len(scores):.2f}")

env.close()