from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_movement_shooting_curriculum")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add curriculum flags.
flag_block = '''        # Test curriculum:
        # movement and shooting are allowed together from the beginning.
        self.curriculum_movement_plus_shooting = True
        self.allow_shooting_in_stage0 = True
        self.allow_use_in_stage0 = False
'''

if "self.curriculum_movement_plus_shooting = True" not in text:
    marker_options = [
        "        self.exit_first_mode = True\n",
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
        "        self.reward_only_mode = True\n",
    ]
    inserted = False
    for marker in marker_options:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break
    if not inserted:
        print("Warning: could not insert curriculum flags.")

# Stop sanitize/tactical code from blocking shoot in early stage.
text = text.replace(
    'reason=shoot_blocked_before_combat_stage',
    'reason=shoot_allowed_movement_plus_shooting'
)

# Common old patterns: shoot -> move_forward before combat.
text = re.sub(
    r'''if\s+action_name\s*==\s*["']shoot["']\s+and\s+[^:\n]*stage[^:\n]*:\s*\n\s*action_name\s*=\s*["']move_forward["']''',
    '''if action_name == "shoot" and False:
            action_name = "move_forward"''',
    text,
    flags=re.DOTALL,
)

text = re.sub(
    r'''if\s+action\s*==\s*["']shoot["']\s+and\s+[^:\n]*stage[^:\n]*:\s*\n\s*action\s*=\s*["']move_forward["']''',
    '''if action == "shoot" and False:
            action = "move_forward"''',
    text,
    flags=re.DOTALL,
)

# Make sure the action names include shoot.
if "shoot" not in text:
    print("Warning: could not verify shoot action in env file.")

p.write_text(text)
print("Patched ViZDoom curriculum to allow movement + shooting.")
