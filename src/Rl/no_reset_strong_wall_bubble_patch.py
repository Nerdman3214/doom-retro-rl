from pathlib import Path
import re

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_no_reset_strong_wall_bubble")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Stronger wall-bubble penalties
# ---------------------------------------------------------
# Old:
#   red wall bubble: -1.5
#   orange wall bubble: -0.4
#
# New test:
#   red wall bubble: -15
#   orange wall bubble: -8
#   pushing into obstacle: -15 through add_penalty
# ---------------------------------------------------------

text = text.replace(
    'reward += self.add_penalty("red_wall_bubble", -1.5)',
    'reward += self.add_penalty("red_wall_bubble", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("orange_wall_bubble", -0.4)',
    'reward += self.add_penalty("orange_wall_bubble", -8.0)'
)

text = text.replace(
    'reward += self.add_penalty("push_into_obstacle", -1.5)',
    'reward += self.add_penalty("push_into_obstacle", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("wall_sensor_forward_blocked", -1.5)',
    'reward += self.add_penalty("wall_sensor_forward_blocked", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("wall_sensor_no_progress_near_wall", -1.0)',
    'reward += self.add_penalty("wall_sensor_no_progress_near_wall", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("push_into_front_wall", -1.0)',
    'reward += self.add_penalty("push_into_front_wall", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("push_wall_forward", -1.5)',
    'reward += self.add_penalty("push_wall_forward", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("wall_contact", -1.0)',
    'reward += self.add_penalty("wall_contact", -15.0)'
)

text = text.replace(
    'reward += self.add_penalty("low_motion_forward", -0.3)',
    'reward += self.add_penalty("low_motion_forward", -15.0)'
)

# ---------------------------------------------------------
# 2. Stop the environment from resetting/terminating on corner trap.
# ---------------------------------------------------------
# Replace:
#   if self.corner_trap_steps >= 120 and not self.sensory_emergency_active:
#       reward -= 5.0
#       terminated = True
#       info["corner_trap_reset"] = True
#       print("[reset] corner trap timeout")
#
# With:
#   big penalty, no termination.
# ---------------------------------------------------------

corner_pattern = r'''        if self\.corner_trap_steps >= 120 and not self\.sensory_emergency_active:
            reward \+= self\.add_penalty\('direct_stuck_penalty', -5\.0\)
            terminated = True
            info\["corner_trap_reset"\] = True
            print\("\[reset\] corner trap timeout"\)
'''

corner_replacement = '''        if self.corner_trap_steps >= 120 and not self.sensory_emergency_active:
            reward += self.add_penalty("corner_trap_no_reset_penalty", -15.0)
            info["corner_trap_penalty"] = True
            print("[no_reset] corner trap penalty only")
'''

text = re.sub(corner_pattern, corner_replacement, text)

# Also handle version that may still say reward -= 5.0.
corner_pattern_2 = r'''        if self\.corner_trap_steps >= 120 and not self\.sensory_emergency_active:
            reward -= 5\.0
            terminated = True
            info\["corner_trap_reset"\] = True
            print\("\[reset\] corner trap timeout"\)
'''

text = re.sub(corner_pattern_2, corner_replacement, text)

# ---------------------------------------------------------
# 3. Stop stuck_counter >= 35 from terminating episode.
# ---------------------------------------------------------
# There are usually two stuck reset blocks. Replace both.
# ---------------------------------------------------------

stuck_pattern = r'''        if self\.stuck_counter >= 35:
            reward -= 2\.0
            terminated = True
            info\["stuck_reset"\] = True
            self\.episode_had_stuck_reset = True
            self\.save_death_review_frames\(reason="stuck_reset"\)
'''

stuck_replacement = '''        if self.stuck_counter >= 35:
            reward += self.add_penalty("stuck_counter_no_reset_penalty", -15.0)
            info["stuck_penalty"] = True
            self.episode_had_stuck_reset = False
            if not self.training_mode:
                self.save_death_review_frames(reason="stuck_penalty")
'''

text = re.sub(stuck_pattern, stuck_replacement, text)

stuck_pattern_2 = r'''        if self\.stuck_counter >= 35:
            reward -= 2\.0
            terminated = True
            info\["stuck_reset"\] = True
            self\.episode_had_stuck_reset = True
'''

text = re.sub(stuck_pattern_2, stuck_replacement, text)

# Handle patched hard-negative version if reward -= 2.0 was already converted.
stuck_pattern_3 = r'''        if self\.stuck_counter >= 35:
            reward \+= self\.add_penalty\('direct_stuck_penalty', -2\.0\)
            terminated = True
            info\["stuck_reset"\] = True
            self\.episode_had_stuck_reset = True
'''

text = re.sub(stuck_pattern_3, stuck_replacement, text)

# ---------------------------------------------------------
# 4. Keep death termination optional.
# ---------------------------------------------------------
# Since Doom auto-resets after death when any key is pressed, we keep death as
# episode termination for PPO bookkeeping, but do NOT add extra manual reset logic here.
#
# The actual reset() function will still be called by Gym after done=True.
# That is okay. What we are removing is artificial stuck/corner resets.
# ---------------------------------------------------------

# ---------------------------------------------------------
# 5. Make sure curriculum does not treat stuck penalty as bad reset.
# ---------------------------------------------------------
text = text.replace(
    'not info.get("stuck_reset", False)',
    'not info.get("stuck_reset", False)'
)

p.write_text(text)
print("Applied no-reset + strong wall bubble patch.")
