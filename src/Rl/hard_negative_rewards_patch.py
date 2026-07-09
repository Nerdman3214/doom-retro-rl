from pathlib import Path
import re

doom_path = Path("env/doom_env.py")
brain_path = Path("rewards/doom_brain_reward.py")

doom = doom_path.read_text()
brain = brain_path.read_text()

# ---------------------------------------------------------
# 1. Back up active files
# ---------------------------------------------------------
doom_path.with_suffix(".py.backup_before_hard_negatives").write_text(doom)
brain_path.with_suffix(".py.backup_before_hard_negatives").write_text(brain)

# ---------------------------------------------------------
# 2. Make add_penalty() convert every penalty to -15.
# This catches most intentional negative rewards in doom_env.py.
# ---------------------------------------------------------
doom = re.sub(
    r"    def add_penalty\(self, name, base_penalty\):.*?\n        return scaled_penalty\n",
    '''    def add_penalty(self, name, base_penalty):
        """
        TEST MODE:
        Convert every intentional negative reward into -15.

        This is intentionally extreme and should only be used for testing
        whether the agent learns to avoid walls/stuck states faster.
        """

        if base_penalty < 0:
            hard_penalty = -15.0
            self.reward_manager.add(name, hard_penalty)
            return hard_penalty

        self.reward_manager.add(name, base_penalty)
        return base_penalty

''',
    doom,
    flags=re.DOTALL,
)

# ---------------------------------------------------------
# 3. Make stagnation penalties -15.
# ---------------------------------------------------------
doom = re.sub(
    r"penalty = min\(1\.0, 0\.10 \* self\.no_position_change_steps\)",
    "penalty = 15.0",
    doom,
)

doom = re.sub(
    r"penalty = min\(0\.75, 0\.05 \* self\.no_distance_progress_steps\)",
    "penalty = 15.0",
    doom,
)

# ---------------------------------------------------------
# 4. Make common direct wall/stuck penalties route through add_penalty
# where possible by replacing hardcoded direct negatives.
# ---------------------------------------------------------
replacements = {
    "reward -= 0.05": "reward += self.add_penalty('direct_small_penalty', -0.05)",
    "reward -= 0.08": "reward += self.add_penalty('direct_small_penalty', -0.08)",
    "reward -= 0.10": "reward += self.add_penalty('direct_small_penalty', -0.10)",
    "reward -= 0.12": "reward += self.add_penalty('direct_small_penalty', -0.12)",
    "reward -= 0.15": "reward += self.add_penalty('direct_small_penalty', -0.15)",
    "reward -= 0.20": "reward += self.add_penalty('direct_small_penalty', -0.20)",
    "reward -= 0.25": "reward += self.add_penalty('direct_small_penalty', -0.25)",
    "reward -= 0.30": "reward += self.add_penalty('direct_small_penalty', -0.30)",
    "reward -= 0.35": "reward += self.add_penalty('direct_small_penalty', -0.35)",
    "reward -= 0.40": "reward += self.add_penalty('direct_small_penalty', -0.40)",
    "reward -= 0.45": "reward += self.add_penalty('direct_small_penalty', -0.45)",
    "reward -= 0.50": "reward += self.add_penalty('direct_small_penalty', -0.50)",
    "reward -= 0.60": "reward += self.add_penalty('direct_small_penalty', -0.60)",
    "reward -= 0.75": "reward += self.add_penalty('direct_small_penalty', -0.75)",
    "reward -= 1.00": "reward += self.add_penalty('direct_wall_penalty', -1.00)",
    "reward -= 1.0": "reward += self.add_penalty('direct_wall_penalty', -1.0)",
    "reward -= 1.50": "reward += self.add_penalty('direct_wall_penalty', -1.50)",
    "reward -= 1.5": "reward += self.add_penalty('direct_wall_penalty', -1.5)",
    "reward -= 2.0": "reward += self.add_penalty('direct_stuck_penalty', -2.0)",
    "reward -= 3.0": "reward += self.add_penalty('direct_wall_penalty', -3.0)",
    "reward -= 4.0": "reward += self.add_penalty('direct_stuck_penalty', -4.0)",
    "reward -= 5.0": "reward += self.add_penalty('direct_stuck_penalty', -5.0)",
    "reward -= 8.0": "reward += self.add_penalty('direct_stuck_penalty', -8.0)",
}

for old, new in replacements.items():
    doom = doom.replace(old, new)

# Keep death penalty harsher than -15 if it already exists.
doom = doom.replace(
    "reward += self.add_penalty('direct_stuck_penalty', -25.0)",
    "reward -= 25.0",
)

# ---------------------------------------------------------
# 5. Patch doom_brain_reward.py direct negatives to -15.
# This affects the shared Doom brain layer.
# ---------------------------------------------------------
brain_replacements = {
    "reward -= 0.001": "reward -= 0.001",
    "reward -= 0.02": "reward -= 15.0",
    "reward -= 0.03": "reward -= 15.0",
    "reward -= 0.04": "reward -= 15.0",
    "reward -= 0.05": "reward -= 15.0",
    "reward -= 0.08": "reward -= 15.0",
    "reward -= 0.15": "reward -= 15.0",
    "reward -= min(0.50, damage_taken * 0.025)": "reward -= 15.0",
}

for old, new in brain_replacements.items():
    brain = brain.replace(old, new)

doom_path.write_text(doom)
brain_path.write_text(brain)

print("Applied hard negative reward test patch.")
print("Backups created:")
print(" - env/doom_env.py.backup_before_hard_negatives")
print(" - rewards/doom_brain_reward.py.backup_before_hard_negatives")
