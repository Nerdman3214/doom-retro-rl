from pathlib import Path
import re

doom_path = Path("env/doom_env.py")
brain_path = Path("rewards/doom_brain_reward.py")

doom = doom_path.read_text()
brain = brain_path.read_text()

doom_backup = Path("env/doom_env.py.backup_before_quad_positive_rewards")
brain_backup = Path("rewards/doom_brain_reward.py.backup_before_quad_positive_rewards")

doom_backup.write_text(doom)
brain_backup.write_text(brain)

print(f"Backup saved to {doom_backup}")
print(f"Backup saved to {brain_backup}")

# ---------------------------------------------------------
# 1. Add a positive reward multiplier helper to DoomEnv
# ---------------------------------------------------------

helper = r'''
    def add_positive(self, name, base_reward):
        """
        TEST MODE:
        Quadruple positive rewards.

        This keeps your hard negative reward experiment intact while making
        successful behavior easier for PPO to notice.

        Example:
        +0.35 -> +1.40
        +3.00 -> +12.00
        +15.00 -> +60.00
        """

        try:
            value = float(base_reward)
        except Exception:
            return base_reward

        if value > 0:
            boosted = value * 4.0
            self.reward_manager.add(name, boosted)
            return boosted

        self.reward_manager.add(name, value)
        return value

'''

if "def add_positive(self, name, base_reward):" not in doom:
    marker = "    def add_penalty("
    if marker not in doom:
        raise SystemExit("Could not find add_penalty() marker in env/doom_env.py")
    doom = doom.replace(marker, helper + "\n" + marker, 1)

# ---------------------------------------------------------
# 2. Replace direct positive reward literals in doom_env.py
# ---------------------------------------------------------
# This converts lines like:
#   reward += 0.35
# into:
#   reward += self.add_positive("direct_positive_reward", 0.35)
#
# It does NOT touch reward += some_function(...) or reward += negative values.
# ---------------------------------------------------------

def replace_direct_positive(match):
    indent = match.group(1)
    value = match.group(2)

    # Avoid changing zero.
    try:
        if float(value) <= 0:
            return match.group(0)
    except Exception:
        return match.group(0)

    return f'{indent}reward += self.add_positive("direct_positive_reward", {value})'

doom = re.sub(
    r'^(\s*)reward\s*\+=\s*([0-9]+(?:\.[0-9]+)?)\b',
    replace_direct_positive,
    doom,
    flags=re.MULTILINE,
)

# ---------------------------------------------------------
# 3. Boost the specific "opposite of wall/stuck bad behavior" rewards harder.
# ---------------------------------------------------------
# Since your negative test made wall/stuck penalties around -15,
# these good recovery actions can become +60 for this experiment.
# ---------------------------------------------------------

specific_replacements = {
    # Half-wall / rail escape.
    'reward += self.add_positive("direct_positive_reward", 0.35)\n                self.reward_manager.add("half_wall_rail_escape_action", 0.35)':
    'reward += self.add_positive("half_wall_rail_escape_action", 15.0)\n                self.reward_manager.add("half_wall_rail_escape_action", 60.0)',

    # Wall sensor escape actions.
    'reward += self.add_positive("direct_positive_reward", 0.15)\n            self.reward_manager.add("wall_sensor_escape_action", 0.15)':
    'reward += self.add_positive("wall_sensor_escape_action", 15.0)\n            self.reward_manager.add("wall_sensor_escape_action", 60.0)',

    # Backing away from wall.
    'reward += self.add_positive("direct_positive_reward", 0.4)\n            self.reward_manager.add("back_away_from_wall", 0.4)':
    'reward += self.add_positive("back_away_from_wall", 15.0)\n            self.reward_manager.add("back_away_from_wall", 60.0)',

    # Escape side walls.
    'reward += self.add_positive("direct_positive_reward", 0.25)\n            self.reward_manager.add("escape_left_wall", 0.25)':
    'reward += self.add_positive("escape_left_wall", 15.0)\n            self.reward_manager.add("escape_left_wall", 60.0)',

    'reward += self.add_positive("direct_positive_reward", 0.25)\n            self.reward_manager.add("escape_right_wall", 0.25)':
    'reward += self.add_positive("escape_right_wall", 15.0)\n            self.reward_manager.add("escape_right_wall", 60.0)',

    # Correct obstacle avoidance.
    'reward += self.add_positive("direct_positive_reward", 0.35)\n                self.reward_manager.add("avoid_obstacle", 0.35)':
    'reward += self.add_positive("avoid_obstacle", 15.0)\n                self.reward_manager.add("avoid_obstacle", 60.0)',

    # Strong stuck escape.
    'reward += self.add_positive("direct_positive_reward", 3.0)\n            self.reward_manager.add("strong_escape_from_stuck", 3.0)':
    'reward += self.add_positive("strong_escape_from_stuck", 15.0)\n            self.reward_manager.add("strong_escape_from_stuck", 60.0)',

    # Boundary escape.
    'reward += self.add_positive("direct_positive_reward", 0.6)\n            self.reward_manager.add("correct_boundary_escape_action", 0.6)':
    'reward += self.add_positive("correct_boundary_escape_action", 15.0)\n            self.reward_manager.add("correct_boundary_escape_action", 60.0)',
}

for old, new in specific_replacements.items():
    doom = doom.replace(old, new)

# ---------------------------------------------------------
# 4. Patch doom_brain_reward.py positives
# ---------------------------------------------------------

if "POSITIVE_REWARD_TEST_MULTIPLIER" not in brain:
    brain = '''POSITIVE_REWARD_TEST_MULTIPLIER = 4.0


def positive_reward(value):
    try:
        value = float(value)
    except Exception:
        return value

    if value > 0:
        return value * POSITIVE_REWARD_TEST_MULTIPLIER

    return value


''' + brain

def replace_brain_positive(match):
    indent = match.group(1)
    value = match.group(2)

    try:
        if float(value) <= 0:
            return match.group(0)
    except Exception:
        return match.group(0)

    return f"{indent}reward += positive_reward({value})"

brain = re.sub(
    r'^(\s*)reward\s*\+=\s*([0-9]+(?:\.[0-9]+)?)\b',
    replace_brain_positive,
    brain,
    flags=re.MULTILINE,
)

# Boost key "good opposite" behavior in doom_brain_reward.py to +60.
brain = brain.replace(
    'reward += positive_reward(0.05)\n            events.append("unstuck_action")',
    'reward += positive_reward(15.0)\n            events.append("unstuck_action")'
)

brain = brain.replace(
    'reward += positive_reward(0.08)\n            events.append("shoot_visible_enemy")',
    'reward += positive_reward(15.0)\n            events.append("shoot_visible_enemy")'
)

brain = brain.replace(
    'reward += positive_reward(0.18)\n            events.append("shoot_centered_enemy")',
    'reward += positive_reward(15.0)\n            events.append("shoot_centered_enemy")'
)

brain = brain.replace(
    'reward += positive_reward(0.20)\n        events.append("picked_item")',
    'reward += positive_reward(15.0)\n        events.append("picked_item")'
)

brain = brain.replace(
    'reward += positive_reward(0.40)\n        events.append("picked_health")',
    'reward += positive_reward(15.0)\n        events.append("picked_health")'
)

brain = brain.replace(
    'reward += positive_reward(0.30)\n        events.append("picked_ammo")',
    'reward += positive_reward(15.0)\n        events.append("picked_ammo")'
)

brain = brain.replace(
    'reward += positive_reward(1.00)\n        events.append("opened_door")',
    'reward += positive_reward(15.0)\n        events.append("opened_door")'
)

doom_path.write_text(doom)
brain_path.write_text(brain)

print("Applied quadruple positive reward test patch.")
print("Important: hard negative rewards are still active.")
print("Specific good opposite behaviors can now receive up to +60.")
