from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_no_corridor_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add config flags in __init__ near rule_enforcer if possible.
flag_block = '''        # Exit-first training:
        # Do not reward fake corridor/route bubbles right now.
        self.reward_route_bubbles = False
        self.reward_secrets_now = False
        self.exit_area_reward = 100.0
        self.exit_progress_scale = 0.02
'''

if "self.reward_route_bubbles = False" not in text:
    marker = "        self.rule_enforcer_reward_scale = 1.0\n"
    if marker in text:
        text = text.replace(marker, marker + flag_block, 1)
    else:
        print("Warning: could not insert exit-first flags.")

# If route progress function returns zone_reward, force non-exit zone_reward to 0.
# This handles common pattern: name, zone_reward, print, return zone_reward.
text = text.replace(
    "return zone_reward",
    "return zone_reward if name == 'level_exit' or getattr(self, 'reward_route_bubbles', True) else 0.0"
)

# Remove secret rewards for now if common names exist.
for old in [
    "reward += secret_reward",
    "reward += 6.0",
    "reward += 5.0",
]:
    # Do not globally replace all +5/+6. Only leave this conservative.
    pass

p.write_text(text)
print("Patched vizdoom_env.py to not reward corridor route bubbles.")
