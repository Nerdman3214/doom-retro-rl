from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_exit_skill_learning_patch")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Import DoomRuleEnforcer if missing
# ---------------------------------------------------------
if "from director.doom_rule_enforcer import DoomRuleEnforcer" not in text:
    text = text.replace(
        "from director.route_director import RouteDirector\n",
        "from director.route_director import RouteDirector\nfrom director.doom_rule_enforcer import DoomRuleEnforcer\n",
        1,
    )

# ---------------------------------------------------------
# 2. Add clean config flags in __init__
# ---------------------------------------------------------
init_marker = "        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())\n"

config_block = """        self.rule_enforcer = DoomRuleEnforcer()
        self.rule_enforcer_reward_scale = 1.0

        # Exit-first skill learning:
        # PPO controls actions. Rewards teach behavior.
        self.exit_first_mode = True
        self.reward_route_bubbles = False
        self.print_route_bubbles = False
        self.reward_secrets_now = False

        # Allow core Doom actions from the start.
        self.allow_shooting_in_stage0 = True
        self.allow_use_in_stage0 = True
        self.curriculum_movement_plus_shooting = True
        self.curriculum_use_doors_early = True

        # Learnable shoot/use rewards.
        self.wasted_shot_penalty = -12.0
        self.poor_aim_shot_penalty = -6.0
        self.good_shot_reward = 12.0
        self.good_use_reward = 10.0
        self.bad_use_spam_penalty = -2.0

"""

if "self.rule_enforcer = DoomRuleEnforcer()" not in text:
    text = text.replace(init_marker, init_marker + config_block, 1)

# ---------------------------------------------------------
# 3. Replace get_stage_config
# ---------------------------------------------------------
stage_config = '''    def get_stage_config(self):
        """
        Curriculum labels only.

        Important:
        We do not hide shoot/use behind stages anymore.
        PPO can choose all core actions from the start.
        """

        configs = {
            0: {
                "name": "exit_skill_learning_start",
                "allow_shoot": True,
                "allow_use": True,
            },
            1: {
                "name": "exit_skill_learning",
                "allow_shoot": True,
                "allow_use": True,
            },
            2: {
                "name": "combat_use_navigation",
                "allow_shoot": True,
                "allow_use": True,
            },
            3: {
                "name": "complete_level_basic",
                "allow_shoot": True,
                "allow_use": True,
            },
            4: {
                "name": "complete_level_plus_secrets_later",
                "allow_shoot": True,
                "allow_use": True,
            },
        }

        return configs.get(self.curriculum_stage, configs[3])


'''

text, n = re.subn(
    r"    def get_stage_config\(self\):.*?(?=\n    def get_allowed_actions\(self\):)",
    stage_config,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing get_stage_config, replacements={n}")

# ---------------------------------------------------------
# 4. Replace get_allowed_actions
# ---------------------------------------------------------
allowed_actions = '''    def get_allowed_actions(self):
        """
        Let the policy learn movement, shooting, and use from the beginning.

        Bad shoot/use decisions are punished by reward, not blocked here.
        """

        return list(self.ACTIONS)
    
'''

text, n = re.subn(
    r"    def get_allowed_actions\(self\):.*?(?=\n    def _angle_diff_degrees\(self, a, b\):)",
    allowed_actions,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing get_allowed_actions, replacements={n}")

# ---------------------------------------------------------
# 5. Disable route/corridor rewards completely
# ---------------------------------------------------------
route_progress = '''    def route_progress_reward(self, game_state):
        """
        Route/corridor bubble rewards disabled.

        We do not reward or print corridor/slope progress.
        Main training target is completing the level.
        """

        return 0.0

'''

text, n = re.subn(
    r"    def route_progress_reward\(self, game_state\):.*?(?=\n    def goal_heading_reward\(self, game_state\):)",
    route_progress,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing route_progress_reward, replacements={n}")

# ---------------------------------------------------------
# 6. Make active target always exit, not route zones
# ---------------------------------------------------------
active_target = '''    def get_active_goal_bubble_target(self, game_state):
        """
        Exit-only target.

        No route/corridor/slope target selection.
        """

        main_goal = self.level_guide.get("main_goal")

        if main_goal is not None:
            return main_goal

        return {
            "name": "level_exit",
            "kind": "exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
        }
    
'''

text, n = re.subn(
    r"    def get_active_goal_bubble_target\(self, game_state\):.*?(?=\n    def route_progress_reward\(self, game_state\):)",
    active_target,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing get_active_goal_bubble_target, replacements={n}")

# ---------------------------------------------------------
# 7. Replace sanitize_action with minimal version
# ---------------------------------------------------------
sanitize = '''    def sanitize_action(self, action_name, game_state):
        """
        Minimal sanitize.

        Do not rewrite shoot/use into movement.
        The reward system should teach whether shoot/use was good or bad.
        """

        if action_name not in self.ACTIONS:
            return "move_forward"

        return action_name

'''

text, n = re.subn(
    r"    def sanitize_action\(self, action_name, game_state\):.*?(?=\n    def _action_name_to_index\(self, action_name\):)",
    sanitize,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing sanitize_action, replacements={n}")

# ---------------------------------------------------------
# 8. Replace update_curriculum_from_progress so it cannot trap Stage 1
# ---------------------------------------------------------
curriculum = '''    def update_curriculum_from_progress(self, game_state):
        """
        Simple curriculum.

        Do not trap the agent in movement-only Stage 1.
        Stages are informational now because all actions are available.
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return

        sx, sy = getattr(self, "spawn_position", (0.0, 0.0))
        dist_from_spawn = ((float(x) - sx) ** 2 + (float(y) - sy) ** 2) ** 0.5

        if self.curriculum_stage == 0 and dist_from_spawn > 128:
            self.curriculum_stage = 1
            print("[viz_curriculum] advanced Stage 0 -> 1")

        elif self.curriculum_stage == 1 and dist_from_spawn > 384:
            self.curriculum_stage = 2
            print("[viz_curriculum] advanced Stage 1 -> 2")

        elif self.curriculum_stage == 2:
            enemy_visible = bool(game_state.get("enemy_visible", False))
            if enemy_visible:
                self.curriculum_stage = 3
                print("[viz_curriculum] advanced Stage 2 -> 3")
    
'''

text, n = re.subn(
    r"    def update_curriculum_from_progress\(self, game_state\):.*?(?=\n    def _state_to_game_state\(self, state\):)",
    curriculum,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Failed replacing update_curriculum_from_progress, replacements={n}")

# ---------------------------------------------------------
# 9. Stop helper override from changing PPO action
# ---------------------------------------------------------
text = text.replace("allow_override=True,", "allow_override=False,")
text = text.replace(
    "if advised_action != action_name:",
    "if False and advised_action != action_name:",
)

# ---------------------------------------------------------
# 10. Remove duplicate route_progress_reward call side effects.
# Since function returns 0 now, this is not critical, but keep code cleaner.
# ---------------------------------------------------------
text = text.replace(
    "        reward += self.route_progress_reward(post_game_state)\n        reward += self.main_goal_progress_reward(director_result)",
    "        reward += self.main_goal_progress_reward(director_result)",
    1,
)

text = text.replace(
    "        reward += self.route_progress_reward(post_game_state)\n        reward += self.goal_attraction_bubble_reward(post_game_state)",
    "        reward += self.goal_attraction_bubble_reward(post_game_state)",
    1,
)

# ---------------------------------------------------------
# 11. Add rule_enforcer reset if missing
# ---------------------------------------------------------
if "self.rule_enforcer.reset()" not in text:
    text = text.replace(
        "        if hasattr(self, \"mission_tracker\"):\n            self.mission_tracker.reset()\n",
        "        if hasattr(self, \"mission_tracker\"):\n            self.mission_tracker.reset()\n        if hasattr(self, \"rule_enforcer\"):\n            self.rule_enforcer.reset()\n",
        1,
    )

# ---------------------------------------------------------
# 12. Add rule_enforcer reward after mission_update if missing
# ---------------------------------------------------------
if "info[\"rule_enforcer\"] = rule_result" not in text:
    marker = "        mission_update = self.mission_tracker.update(post_game_state, world_state)\n"
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

        # Early use/door reward.
        try:
            door_visible = bool(post_game_state.get("door_visible", False))
            door_centered = bool(post_game_state.get("door_centered", False))
            near_use_point = bool(post_game_state.get("near_use_point", False))
            opened_door = bool(post_game_state.get("opened_door", False))
            scene_label = post_game_state.get("scene_label")

            door_like = (
                door_visible
                or door_centered
                or near_use_point
                or scene_label in ["door_or_button", "switch", "locked_door"]
            )

            if action_name == "use":
                if door_like:
                    reward += float(getattr(self, "good_use_reward", 10.0))
                    info["use_quality"] = "good_near_door"
                else:
                    reward += float(getattr(self, "bad_use_spam_penalty", -2.0))
                    info["use_quality"] = "spam"

            if opened_door:
                reward += 25.0
                info["opened_door"] = True
        except Exception:
            pass

        # Shooting reward discipline.
        try:
            enemy_visible = bool(post_game_state.get("enemy_visible", False))
            enemy_centered = bool(post_game_state.get("enemy_centered", False))
            ammo = int(post_game_state.get("ammo", 0) or 0)

            if action_name == "shoot":
                if enemy_visible and enemy_centered and ammo > 0:
                    reward += float(getattr(self, "good_shot_reward", 12.0))
                    info["shot_quality"] = "good_centered"
                elif enemy_visible and ammo > 0:
                    reward += float(getattr(self, "poor_aim_shot_penalty", -6.0))
                    info["shot_quality"] = "poor_aim"
                else:
                    reward += float(getattr(self, "wasted_shot_penalty", -12.0))
                    info["shot_quality"] = "wasted"
        except Exception:
            pass

        if self.step_count % 100 == 0:
            print(
                f"[viz_action] step={self.step_count} "
                f"action={action_name} "
                f"buttons={buttons} "
                f"stage={self.curriculum_stage}"
            )

'''
    text = text.replace(marker, marker + insert, 1)

p.write_text(text)
print("Applied ViZDoom exit-skill-learning patch.")
