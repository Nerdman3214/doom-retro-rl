import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.doom_env import DoomEnv
import numpy as np

# Load the latest model
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
latest_checkpoint = os.path.join(CHECKPOINT_DIR, "doom_rl_agent_280000_steps.zip")

env = DoomEnv(launch_doom=False, record=False)

model = PPO.load(latest_checkpoint, env=env)

# Evaluate for 10 episodes
episodes = 10
total_rewards = []

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < 1000:  # limit steps to avoid infinite loops
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
    total_rewards.append(episode_reward)
    print(f"Episode {ep+1}: Reward = {episode_reward:.4f}, Steps = {steps}")

avg_reward = np.mean(total_rewards)
print(f"\nAverage reward over {episodes} episodes: {avg_reward:.4f}")

# Random baseline
print("\nTesting random agent...")
random_rewards = []
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < 1000:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
    random_rewards.append(episode_reward)

random_avg = np.mean(random_rewards)
print(f"Random average reward: {random_avg:.4f}")

if avg_reward > random_avg:
    print("Agent performs better than random!")
else:
    print("Agent does not perform better than random.")