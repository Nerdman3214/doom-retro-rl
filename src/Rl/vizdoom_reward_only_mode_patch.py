from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_reward_only_mode")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add reward-only mode flags in __init__
# ---------------------------------------------------------

init_marker_candidates = [
    "        self.last_reward_debug = {}\n",
    "        self.previous_game_state = None\n",
    "        self.route_progress_level = 0\n",
]

flag_block = '''        # -----------------------------------------------------
        # Reward-only training mode
        # -----------------------------------------------------
        # In this mode, PPO chooses the action.
        # Helpers may add rewards/debug info, but they must not replace actions.
        self.reward_only_mode = True
        self.disable_action_helpers = True
        self.disable_fast_enemy_reaction = True
        self.disable_recovery_action_override = True
        self.disable_advisor_action_override = True
        self.disable_route_action_override = True
'''

if "self.reward_only_mode = True" not in text:
    inserted = False
    for marker in init_marker_candidates:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break
    if not inserted:
        print("Warning: could not find __init__ marker for reward-only flags.")

# ---------------------------------------------------------
# 2. Make common action-helper methods advisory-only
# ---------------------------------------------------------
# This rewrites known helper methods so they return the PPO action unchanged.
# If a method is not found, it is skipped safely.
# ---------------------------------------------------------

def replace_method(source, method_name, body):
    pattern = rf'''    def {method_name}\(self,.*?(?=\n    def |\nclass |\Z)'''
    new_source, count = re.subn(pattern, body.rstrip() + "\n\n", source, flags=re.DOTALL)
    if count:
        print(f"Replaced {method_name} with advisory-only version. count={count}")
    else:
        print(f"Skipped {method_name}; not found.")
    return new_source

advisory_methods = {
    "fast_enemy_reaction_action": '''    def fast_enemy_reaction_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not override PPO action.
        Combat behavior should be learned through rewards.
        """
        return action
''',

    "fast_enemy_reaction": '''    def fast_enemy_reaction(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not override PPO action.
        """
        return action
''',

    "advisor_action": '''    def advisor_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Advisor may be used for debug/reward later, but not action replacement.
        """
        return action
''',

    "action_advisor": '''    def action_advisor(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Advisor may be used for debug/reward later, but not action replacement.
        """
        return action
''',

    "recovery_action": '''    def recovery_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Recovery should be reward/debug only during PPO training.
        """
        return action
''',

    "hard_recovery_action": '''    def hard_recovery_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not force recovery action.
        """
        return action
''',

    "wall_assist_action": '''    def wall_assist_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not force wall helper action.
        """
        return action
''',

    "route_assist_action": '''    def route_assist_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not force route helper action.
        """
        return action
''',
}

for name, body in advisory_methods.items():
    text = replace_method(text, name, body)

# ---------------------------------------------------------
# 3. Neutralize obvious inline fast_enemy_reaction overrides
# ---------------------------------------------------------
# Example old logs:
#   [vizdoom] fast_enemy_reaction: turn_left -> move_backward
#
# This patch keeps logs/reward possible, but prevents replacing action.
# ---------------------------------------------------------

text = re.sub(
    r'''(\s*)before_fast_enemy\s*=\s*action\s*\n\s*action\s*=\s*self\.fast_enemy_reaction[^\n]*\n''',
    r'''\1before_fast_enemy = action
\1# reward_only_mode: fast enemy reaction is advisory only
\1# action remains PPO-selected
''',
    text,
)

text = re.sub(
    r'''(\s*)before\s*=\s*action\s*\n\s*action\s*=\s*self\.fast_enemy_reaction[^\n]*\n''',
    r'''\1before = action
\1# reward_only_mode: fast enemy reaction is advisory only
\1# action remains PPO-selected
''',
    text,
)

# ---------------------------------------------------------
# 4. If code has actual_action/helper_action, keep actual_action as action.
# ---------------------------------------------------------

text = re.sub(
    r'''actual_action\s*=\s*helper_action''',
    '''actual_action = action  # reward_only_mode: do not use helper_action''',
    text,
)

text = re.sub(
    r'''action\s*=\s*helper_action''',
    '''action = action  # reward_only_mode: ignore helper_action''',
    text,
)

# ---------------------------------------------------------
# 5. Make sure mission_tracker exists if doom_brain reward is still called.
# ---------------------------------------------------------

import_line = "from navigation.mission_plan import MissionTracker, get_freedoom1_e1m1_mission\n"

if import_line not in text:
    marker = "from navigation.level_guides import get_level_guide\n"
    if marker in text:
        text = text.replace(marker, marker + import_line, 1)
    else:
        text = import_line + text

mission_init = '''        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())
'''

if "self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())" not in text:
    marker = "        self.level_guide = get_level_guide(self.current_level_name)\n"
    if marker in text:
        text = text.replace(marker, marker + mission_init, 1)
    else:
        print("Warning: could not find level_guide marker for mission_tracker.")

mission_reset = '''        if hasattr(self, "mission_tracker"):
            self.mission_tracker.reset()
'''

if "self.mission_tracker.reset()" not in text:
    reset_pos = text.find("    def reset(")
    if reset_pos != -1:
        before = text[:reset_pos]
        after = text[reset_pos:]
        reset_marker_candidates = [
            "        self.step_count = 0\n",
            "        self.current_step = 0\n",
            "        self.route_progress_level = 0\n",
        ]
        inserted = False
        for marker in reset_marker_candidates:
            if marker in after:
                after = after.replace(marker, marker + mission_reset, 1)
                inserted = True
                break
        text = before + after
        if not inserted:
            print("Warning: could not insert mission_tracker.reset().")

p.write_text(text)
print("Applied ViZDoom reward-only mode patch.")
