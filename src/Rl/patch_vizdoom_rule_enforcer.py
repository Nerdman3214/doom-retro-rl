from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_rule_enforcer")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Import rule enforcer
# ---------------------------------------------------------
import_line = "from director.doom_rule_enforcer import DoomRuleEnforcer\n"

if import_line not in text:
    marker = "from navigation.mission_plan import MissionTracker, get_freedoom1_e1m1_mission\n"
    if marker in text:
        text = text.replace(marker, marker + import_line, 1)
    else:
        text = import_line + text

# ---------------------------------------------------------
# 2. Add self.rule_enforcer in __init__
# ---------------------------------------------------------
init_line = '''        self.rule_enforcer = DoomRuleEnforcer()
        self.rule_enforcer_reward_scale = 1.0
        self.teacher_action_override = False
'''

if "self.rule_enforcer = DoomRuleEnforcer()" not in text:
    markers = [
        "        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())\n",
        "        self.current_step = 0\n",
        "        self.step_count = 0\n",
    ]

    inserted = False
    for marker in markers:
        if marker in text:
            text = text.replace(marker, marker + init_line, 1)
            inserted = True
            break

    if not inserted:
        raise SystemExit("Could not insert rule_enforcer in __init__.")

# ---------------------------------------------------------
# 3. Reset rule enforcer
# ---------------------------------------------------------
reset_line = '''        if hasattr(self, "rule_enforcer"):
            self.rule_enforcer.reset()
'''

if "self.rule_enforcer.reset()" not in text:
    reset_pos = text.find("    def reset(")

    if reset_pos != -1:
        before = text[:reset_pos]
        after = text[reset_pos:]

        reset_markers = [
            "        self.mission_tracker.reset()\n",
            "        self.current_step = 0\n",
            "        self.step_count = 0\n",
            "        self.route_progress_level = 0\n",
        ]

        inserted = False
        for marker in reset_markers:
            if marker in after:
                after = after.replace(marker, marker + reset_line, 1)
                inserted = True
                break

        text = before + after

        if not inserted:
            print("Warning: could not find reset marker for rule_enforcer.reset().")

# ---------------------------------------------------------
# 4. Add rule enforcer reward after mission_update exists
# ---------------------------------------------------------
# Most versions already have this line from doom_brain integration:
# mission_update = self.mission_tracker.update(post_game_state, world_state)
# Insert immediately after it.
# ---------------------------------------------------------

needle = "        mission_update = self.mission_tracker.update(post_game_state, world_state)\n"

insert = '''        rule_result = self.rule_enforcer.evaluate(
            game_state=post_game_state,
            action_name=action_name,
            mission_update=mission_update,
            level_guide=getattr(self, "level_guide", None),
        )

        reward += float(rule_result.get("reward", 0.0)) * float(
            getattr(self, "rule_enforcer_reward_scale", 1.0)
        )

        info["rule_enforcer"] = rule_result

        if self.current_step % 50 == 0:
            print(
                "[rule_enforcer] "
                f"preferred={rule_result.get('preferred_action')} "
                f"action={action_name} "
                f"reward={rule_result.get('reward'):.2f} "
                f"events={rule_result.get('events', [])[:4]} "
                f"target={rule_result.get('target')}"
            )

        # Optional teacher mode.
        # Keep False for PPO reward-only training.
        if getattr(self, "teacher_action_override", False):
            preferred = rule_result.get("preferred_action")
            if preferred in getattr(self, "action_names", []):
                action_name = preferred
'''

if insert not in text:
    if needle in text:
        text = text.replace(needle, needle + insert, 1)
    else:
        print("Warning: mission_update marker not found. Rule enforcer reward not inserted.")

p.write_text(text)
print("Patched VizDoomEnv with DoomRuleEnforcer.")
