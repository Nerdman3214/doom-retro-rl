from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_allow_shoot_after_post_corridor")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Make sanitize_action stop blocking shoot once post_corridor has been reached.
# This patch is intentionally defensive and looks for common "shoot not allowed" patterns.
replacements = 0

patterns = [
    (
        '''if action_name == "shoot" and "shoot" not in allowed_actions:''',
        '''if (
            action_name == "shoot"
            and "shoot" not in allowed_actions
            and not getattr(self, "stage01_post_corridor_reached", False)
        ):'''
    ),
    (
        '''if action_name == "shoot" and not self.shoot_allowed:''',
        '''if (
            action_name == "shoot"
            and not self.shoot_allowed
            and not getattr(self, "stage01_post_corridor_reached", False)
        ):'''
    ),
]

for old, new in patterns:
    if old in text:
        text = text.replace(old, new)
        replacements += 1

# If allowed_actions is built as a list, append shoot after post_corridor.
marker = '''        return allowed_actions
'''
insert = '''        if getattr(self, "stage01_post_corridor_reached", False) and "shoot" not in allowed_actions:
            allowed_actions.append("shoot")

'''

if marker in text and "stage01_post_corridor_reached\", False) and \"shoot\" not in allowed_actions" not in text:
    text = text.replace(marker, insert + marker, 1)
    replacements += 1

if replacements == 0:
    print("No generic shoot-block pattern found. Paste get_allowed_actions() and sanitize_action() if this does not change behavior.")
else:
    print(f"Applied shoot-after-post-corridor changes: {replacements}")

p.write_text(text)
