from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_allow_shoot_use_early")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add explicit flags.
flag_block = '''        # Skill test:
        # Let PPO learn movement + shooting + use from the start.
        self.allow_shooting_in_stage0 = True
        self.allow_use_in_stage0 = True
        self.curriculum_movement_plus_shooting = True
        self.curriculum_use_doors_early = True
'''

if "self.curriculum_use_doors_early = True" not in text:
    markers = [
        "        self.exit_first_mode = True\n",
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
        "        self.reward_only_mode = True\n",
    ]

    inserted = False
    for marker in markers:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break

    if not inserted:
        print("Warning: could not insert shoot/use curriculum flags.")

# Remove direct action rewrite blocks for shoot/use in early curriculum.
# These regexes turn blocks like `if action_name == "use" and stage...: action_name = "move_forward"` into disabled blocks.
patterns = [
    r'''if\s+action_name\s*==\s*["']shoot["'][^:\n]*:\s*\n\s*action_name\s*=\s*["']move_forward["']''',
    r'''if\s+action_name\s*==\s*["']use["'][^:\n]*:\s*\n\s*action_name\s*=\s*["']move_forward["']''',
    r'''if\s+action\s*==\s*["']shoot["'][^:\n]*:\s*\n\s*action\s*=\s*["']move_forward["']''',
    r'''if\s+action\s*==\s*["']use["'][^:\n]*:\s*\n\s*action\s*=\s*["']move_forward["']''',
]

for pat in patterns:
    text = re.sub(
        pat,
        "if False:\n            pass  # shoot/use allowed early",
        text,
        flags=re.DOTALL,
    )

# Replace common sanitize log reasons so we can grep them.
text = text.replace("shoot_blocked_before_combat_stage", "shoot_allowed_stage0")
text = text.replace("use_blocked_before_use_stage", "use_allowed_stage0")
text = text.replace("use -> move_forward", "use_allowed_no_rewrite")

# Direct penalties should be old-ish, not 6x harsh.
text = text.replace("self.wasted_shot_penalty = -60.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.wasted_shot_penalty = -24.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.poor_aim_shot_penalty = -24.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.poor_aim_shot_penalty = -12.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.good_shot_reward = 18.0", "self.good_shot_reward = 12.0")
text = text.replace("self.good_shot_reward = 8.0", "self.good_shot_reward = 12.0")

p.write_text(text)
print("Patched ViZDoom to allow shoot/use early and reduce harsh shot penalties.")
