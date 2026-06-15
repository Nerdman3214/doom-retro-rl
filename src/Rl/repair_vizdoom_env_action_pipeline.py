from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_action_pipeline_repair")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Repair corrupted main_goal_progress_reward()
# ---------------------------------------------------------

main_goal_progress_clean = r'''    def main_goal_progress_reward(self, director_result):
        """
        Reward getting closer to the real E1M1 exit.

        Exit-first mode:
        no corridor/slope rewards.
        """

        if director_result is None:
            return 0.0

        distance = director_result.get("distance")
        distance_delta = float(director_result.get("distance_delta", 0.0) or 0.0)
        hint = director_result.get("hint")

        if distance is None:
            return 0.0

        reward = 0.0

        if not hasattr(self, "best_main_goal_distance"):
            self.best_main_goal_distance = None

        if self.best_main_goal_distance is None:
            self.best_main_goal_distance = float(distance)

        if distance_delta > 1.0:
            progress_reward = min(0.12, distance_delta / 96.0)
            reward += progress_reward
            self.reward_manager.add("main_goal_closer", progress_reward)

        if float(distance) < self.best_main_goal_distance - 8.0:
            self.best_main_goal_distance = float(distance)
            reward += 0.10
            self.reward_manager.add("main_goal_new_best", 0.10)

        if distance_delta < -6.0:
            penalty = min(0.12, abs(distance_delta) / 96.0)
            reward -= penalty
            self.reward_manager.add("main_goal_moving_away", -penalty)

        if hint == "recover_unstuck":
            reward -= 0.08
            self.reward_manager.add("main_goal_stuck", -0.08)

        if hint == "goal_reached":
            reward += 10.0
            self.reward_manager.add("main_goal_reached", 10.0)

        return reward

'''

text, count = re.subn(
    r"    def main_goal_progress_reward\(self, director_result\):.*?(?=\n    def goal_progress_reward\(self, director_result\):)",
    main_goal_progress_clean,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace corrupted main_goal_progress_reward once, replaced {count}")

# ---------------------------------------------------------
# 2. Make shoot/use available from Stage 0.
# ---------------------------------------------------------

get_stage_config_clean = r'''    def get_stage_config(self):
        configs = {
            0: {
                "name": "movement_combat_use_basic",
                "allow_shoot": True,
                "allow_use": True,
            },
            1: {
                "name": "movement_combat_use_escape",
                "allow_shoot": True,
                "allow_use": True,
            },
            2: {
                "name": "doors_combat_use",
                "allow_shoot": True,
                "allow_use": True,
            },
            3: {
                "name": "complete_level_basic",
                "allow_shoot": True,
                "allow_use": True,
            },
        }

        return configs.get(self.curriculum_stage, configs[3])


'''

text, count = re.subn(
    r"    def get_stage_config\(self\):.*?(?=\n    def get_allowed_actions\(self\):)",
    get_stage_config_clean,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace get_stage_config once, replaced {count}")

get_allowed_actions_clean = r'''    def get_allowed_actions(self):
        """
        Exit-first skill learning:
        PPO can choose movement, shooting, and use from the start.

        We do not hide shoot/use by curriculum anymore.
        Bad shoot/use choices are handled by reward, not action blocking.
        """

        return list(self.ACTIONS)
    
'''

text, count = re.subn(
    r"    def get_allowed_actions\(self\):.*?(?=\n    def _angle_diff_degrees\(self, a, b\):)",
    get_allowed_actions_clean,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace get_allowed_actions once, replaced {count}")

# ---------------------------------------------------------
# 3. Make sanitize_action stop rewriting shoot/use.
# ---------------------------------------------------------

sanitize_clean = r'''    def sanitize_action(self, action_name, game_state):
        """
        Minimal safety sanitize.

        Do NOT rewrite shoot/use into movement.
        If the agent wastes ammo or presses use randomly, reward handles it.
        """

        if action_name not in self.ACTIONS:
            return "move_forward"

        return action_name

'''

text, count = re.subn(
    r"    def sanitize_action\(self, action_name, game_state\):.*?(?=\n    def _action_name_to_index\(self, action_name\):)",
    sanitize_clean,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace sanitize_action once, replaced {count}")

# ---------------------------------------------------------
# 4. Make sure flags do not contradict each other.
# ---------------------------------------------------------

text = text.replace("self.allow_use_in_stage0 = False", "self.allow_use_in_stage0 = True")
text = text.replace("self.allow_shooting_in_stage0 = False", "self.allow_shooting_in_stage0 = True")

# ---------------------------------------------------------
# 5. Disable shared helper override in ViZDoom.
# ---------------------------------------------------------

text = text.replace(
    "allow_override=True,",
    "allow_override=False,",
)

text = text.replace(
    "if advised_action != action_name:",
    "if False and advised_action != action_name:",
)

# ---------------------------------------------------------
# 6. Remove late shooting exploration block.
# It happens after make_action(), so it cannot actually press shoot.
# ---------------------------------------------------------

text, count = re.subn(
    r"\n        # -----------------------------------------------------\n"
    r"        # Enemy-visible shooting exploration\n"
    r".*?"
    r"\n\n        rule_result = self\.rule_enforcer\.evaluate\(",
    "\n        rule_result = self.rule_enforcer.evaluate(",
    text,
    flags=re.DOTALL,
)

print(f"Removed late shooting exploration blocks: {count}")

# ---------------------------------------------------------
# 7. Keep old-ish stable negative settings.
# ---------------------------------------------------------

text = text.replace("self.wasted_shot_penalty = -60.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.wasted_shot_penalty = -24.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.poor_aim_shot_penalty = -24.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.poor_aim_shot_penalty = -12.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.good_shot_reward = 18.0", "self.good_shot_reward = 12.0")
text = text.replace("self.good_shot_reward = 8.0", "self.good_shot_reward = 12.0")

# ---------------------------------------------------------
# 8. Add clean action debug so we can prove actual button mapping.
# ---------------------------------------------------------

debug_insert = r'''        if self.step_count % 100 == 0:
            print(
                f"[viz_action] step={self.step_count} "
                f"action={action_name} "
                f"buttons={buttons} "
                f"stage={self.curriculum_stage}"
            )

'''

if "[viz_action]" not in text:
    marker = "        try:\n            self.game.make_action(buttons, self.frame_skip)\n"
    if marker in text:
        text = text.replace(marker, debug_insert + marker, 1)
    else:
        print("Warning: could not insert viz_action debug.")

p.write_text(text)
print("Repaired ViZDoom action pipeline.")
