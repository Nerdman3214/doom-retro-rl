from pathlib import Path
import re

# ---------------------------------------------------------
# 1. Patch rule enforcer: always target level exit
# ---------------------------------------------------------

p = Path("director/doom_rule_enforcer.py")
text = p.read_text()

backup = Path("director/doom_rule_enforcer.py.backup_before_no_corridor_targets")
backup.write_text(text)
print(f"Backup saved to {backup}")

pattern = r'''    def _current_target\(self, game_state, mission_update=None, level_guide=None\):
.*?
        return None
'''

replacement = '''    def _current_target(self, game_state, mission_update=None, level_guide=None):
        """
        Exit-first mode.

        Do not use corridor, slope, post-corridor-room, route bubbles,
        or secret targets. The agent should adapt from experience, not be
        told that an early corridor zone is progress.

        Only target: real E1M1 exit area.
        """

        return {
            "name": "level_exit",
            "kind": "exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
            "hint": "complete_level",
        }
'''

text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)

if count != 1:
    raise SystemExit(f"Expected to replace _current_target once, replaced {count}")

# Remove optional objective reward from secrets/advanced for now.
text = text.replace(
    '''        if mission_update:
            for event in mission_update.get("events", []):
                if "optional_reached" in str(event):
                    reward += 15.0
                    events.append("optional_objective_reached")

''',
    '''        # Secrets/optional objectives disabled for exit-first training.
        # Add them back after the agent can complete E1M1.

'''
)

p.write_text(text)
print("Patched DoomRuleEnforcer: exit-only target, no corridor/slope/secret targets.")


# ---------------------------------------------------------
# 2. Patch vizdoom_env.py: suppress route-progress logs/rewards except exit
# ---------------------------------------------------------

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_no_corridor_route_system")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add/force flags.
flag_block = '''        # Absolute exit-first mode:
        # no route bubble reward, no route bubble print, no corridor/slope targets.
        self.exit_first_mode = True
        self.reward_route_bubbles = False
        self.print_route_bubbles = False
        self.reward_secrets_now = False
        self.exit_area_reward = 100.0
'''

if "self.exit_first_mode = True" not in text:
    marker_options = [
        "        self.rule_enforcer_reward_scale = 1.0\n",
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
    ]

    inserted = False
    for marker in marker_options:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break

    if not inserted:
        print("Warning: could not insert exit-first flags.")

# Silence route-progress print blocks.
# This keeps the function alive but prevents misleading console messages.
text = re.sub(
    r'''            print\(
                f"\[viz_route_progress\] reached=\{name\} "
                f"level=\{self\.route_progress_level\} "
                f"x=\{x:\.1f\} y=\{y:\.1f\} "
                f"reward=\{zone_reward:\.2f\}"
            \)
''',
    '''            if getattr(self, "print_route_bubbles", False) or name == "level_exit":
                print(
                    f"[viz_route_progress] reached={name} "
                    f"level={self.route_progress_level} "
                    f"x={x:.1f} y={y:.1f} "
                    f"reward={zone_reward:.2f}"
                )
''',
    text,
    flags=re.DOTALL,
)

# If the exact print block did not match, use a broader line-based guard.
if "[viz_route_progress]" in text and "print_route_bubbles" not in text[text.find("[viz_route_progress]")-300:text.find("[viz_route_progress]")+500]:
    text = text.replace(
        'print(\n                f"[viz_route_progress] reached={name} "',
        'if getattr(self, "print_route_bubbles", False) or name == "level_exit":\n                print(\n                    f"[viz_route_progress] reached={name} "',
    )

# Force route bubble rewards to zero unless it is the actual exit.
text = text.replace(
    "return zone_reward if name == 'level_exit' or getattr(self, 'reward_route_bubbles', True) else 0.0",
    "return zone_reward if name == 'level_exit' else 0.0",
)

text = text.replace(
    "return zone_reward",
    "return zone_reward if name == 'level_exit' else 0.0",
)

# Stop mission tracker route targets from driving the rule enforcer.
# Rule enforcer is now exit-only, but this also keeps info cleaner.
text = text.replace(
    "mission_update = self.mission_tracker.update(post_game_state, world_state)",
    "mission_update = {'target': {'name': 'level_exit', 'kind': 'exit', 'x': -400.0, 'y': 1296.0, 'radius': 160.0}, 'events': []}"
)

p.write_text(text)
print("Patched VizDoomEnv: no corridor/slope route prints or targets.")


# ---------------------------------------------------------
# 3. Patch mission_plan.py: make E1M1 mission exit-only
# ---------------------------------------------------------

p = Path("navigation/mission_plan.py")
text = p.read_text()

backup = Path("navigation/mission_plan.py.backup_before_exit_only")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Replace get_freedoom1_e1m1_mission with an exit-only version if present.
pattern = r'''def get_freedoom1_e1m1_mission\(\):
.*?(?=\ndef |\nclass |\Z)'''

replacement = '''def get_freedoom1_e1m1_mission():
    """
    Exit-only E1M1 mission.

    No corridor/slope/secret objectives.
    The agent should learn/adapt through rewards:
    - survival
    - combat enough to stay alive
    - moving closer to the actual exit
    - entering exit area
    - completing the level
    """

    return MissionPlan(
        name="freedoom1_e1m1_exit_only",
        objectives=[
            MissionObjective(
                name="level_exit",
                kind="exit",
                x=-400.0,
                y=1296.0,
                radius=160.0,
                reward=100.0,
                required=True,
                hint="complete_level",
            ),
        ],
        final_goal={
            "name": "level_exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
        },
    )

'''

text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)

if count != 1:
    print(f"Warning: get_freedoom1_e1m1_mission replacement count={count}. Function may use different format.")

p.write_text(text)
print("Patched mission_plan.py: exit-only mission.")
