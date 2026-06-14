from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_combat_shoot_prior_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Make the advisor matter more during combat.
text = re.sub(
    r"self\.action_prior_reward_scale\s*=\s*[0-9.]+",
    "self.action_prior_reward_scale = 0.08",
    text,
)

text = re.sub(
    r"self\.action_prior_min_confidence\s*=\s*[0-9.]+",
    "self.action_prior_min_confidence = 0.35",
    text,
)

# Strengthen mismatch penalty only when the advisor strongly says shoot.
old = '''return -0.002'''
new = '''return -0.02 if str(advised_action) == "shoot" and float(confidence) >= 0.50 else -0.002'''

if old in text and "return -0.02 if str(advised_action) == \"shoot\"" not in text:
    text = text.replace(old, new, 1)

p.write_text(text)
print("Patched stronger combat shoot-prior reward.")
