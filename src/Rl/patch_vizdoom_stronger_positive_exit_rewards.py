from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_stronger_positive_exit_rewards")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Strengthen main_goal_progress_reward.
text = text.replace(
    "progress_reward = min(0.12, distance_delta / 96.0)",
    "progress_reward = min(5.0, distance_delta / 12.0)",
)

text = text.replace(
    'self.reward_manager.add("main_goal_closer", progress_reward)',
    'self.reward_manager.add("main_goal_closer_strong", progress_reward)',
)

text = text.replace(
    "reward += 0.10\n            self.reward_manager.add(\"main_goal_new_best\", 0.10)",
    "reward += 15.0\n            self.reward_manager.add(\"main_goal_new_best\", 15.0)",
)

text = text.replace(
    "reward += 10.0\n            self.reward_manager.add(\"main_goal_reached\", 10.0)",
    "reward += 300.0\n            self.reward_manager.add(\"main_goal_reached\", 300.0)",
)

# Strengthen exit_distance_progress_reward.
text = text.replace(
    "reward += min(0.08, improvement / 128.0)",
    "reward += min(4.0, improvement / 16.0)",
)

text = text.replace(
    "reward += 0.10\n            self.reward_manager.add(\"new_best_exit_distance\", 0.10)",
    "reward += 12.0\n            self.reward_manager.add(\"new_best_exit_distance\", 12.0)",
)

# Strong timeout penalty before return.
old = "        return obs, float(reward), bool(done), bool(truncated), info\n"

new = '''        if truncated and not done:
            reward -= 300.0
            info["timeout_penalty"] = -300.0

        return obs, float(reward), bool(done), bool(truncated), info
'''

if old not in text:
    raise SystemExit("Could not find final return line to add timeout penalty")

text = text.replace(old, new, 1)

p.write_text(text)
print("Strengthened positive exit rewards and added timeout penalty.")
