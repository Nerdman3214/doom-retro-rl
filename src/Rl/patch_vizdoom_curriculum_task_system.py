from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_curriculum_task_system")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Import curriculum config
# ---------------------------------------------------------
if "from curriculum.task_config import get_task_config" not in text:
    text = text.replace(
        "from env.shared_doom_logic import SharedDoomLogic\n",
        "from env.shared_doom_logic import SharedDoomLogic\nfrom curriculum.task_config import get_task_config\n",
        1,
    )

# ---------------------------------------------------------
# 2. Add task config in __init__
# ---------------------------------------------------------
marker = "        self.current_level_name = \"freedoom1_e1m1\"\n        self.level_guide = get_level_guide(self.current_level_name)\n"

insert = """        # -----------------------------------------------------
        # Small-task curriculum
        # -----------------------------------------------------
        # Pick task using:
        #   VIZDOOM_TASK=stage_01_navigation python training/train_vizdoom_agent.py
        self.task_name = os.environ.get("VIZDOOM_TASK", "stage_01_navigation")
        self.task_config = get_task_config(self.task_name)
        print(f"[viz_task] active={self.task_config.name}: {self.task_config.description}")

"""

if "self.task_config = get_task_config" not in text:
    if marker not in text:
        raise SystemExit("Could not find level_guide marker")
    text = text.replace(marker, marker + insert, 1)

# ---------------------------------------------------------
# 3. Replace get_stage_config
# ---------------------------------------------------------
stage_config = '''    def get_stage_config(self):
        """
        Stage label comes from the curriculum task.
        Action/reward control is task-based, not hardcoded stage-based.
        """

        return {
            "name": getattr(self.task_config, "name", "unknown_task"),
            "allow_shoot": "shoot" in self.task_config.allowed_actions,
            "allow_use": "use" in self.task_config.allowed_actions,
        }


'''

text, n = re.subn(
    r"    def get_stage_config\(self\):.*?(?=\n    def get_allowed_actions\(self\):)",
    stage_config,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Expected to replace get_stage_config once, replaced {n}")

# ---------------------------------------------------------
# 4. Replace get_allowed_actions
# ---------------------------------------------------------
allowed = '''    def get_allowed_actions(self):
        """
        Current task decides which actions are available.

        This copies the common ViZDoom pattern:
        one skill task = one clear action set.
        """

        allowed = list(getattr(self.task_config, "allowed_actions", self.ACTIONS))
        return [a for a in allowed if a in self.ACTIONS]
    
'''

text, n = re.subn(
    r"    def get_allowed_actions\(self\):.*?(?=\n    def _angle_diff_degrees\(self, a, b\):)",
    allowed,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Expected to replace get_allowed_actions once, replaced {n}")

# ---------------------------------------------------------
# 5. Disable route bubble progress reward/prints
# ---------------------------------------------------------
route_func = '''    def route_progress_reward(self, game_state):
        """
        Route/corridor bubble rewards disabled by default.

        Small-task curriculum uses task targets instead of loose route bubbles.
        This avoids duplicate corridor_left_turn / sloped_corridor_mid farming.
        """

        if getattr(self.task_config, "enable_route_bubbles", False):
            return 0.0

        return 0.0

'''

text, n = re.subn(
    r"    def route_progress_reward\(self, game_state\):.*?(?=\n    def goal_heading_reward\(self, game_state\):)",
    route_func,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Expected to replace route_progress_reward once, replaced {n}")

# ---------------------------------------------------------
# 6. Make active goal bubble target use current task target
# ---------------------------------------------------------
active_target = '''    def get_active_goal_bubble_target(self, game_state):
        """
        Current curriculum task target.

        This replaces loose route-zone target selection.
        """

        return {
            "name": self.task_config.target_name,
            "kind": "task_target",
            "x": float(self.task_config.target_x),
            "y": float(self.task_config.target_y),
            "radius": float(self.task_config.target_radius),
        }
    
'''

text, n = re.subn(
    r"    def get_active_goal_bubble_target\(self, game_state\):.*?(?=\n    def route_progress_reward\(self, game_state\):)",
    active_target,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Expected to replace get_active_goal_bubble_target once, replaced {n}")

# ---------------------------------------------------------
# 7. Replace sanitize_action
# ---------------------------------------------------------
sanitize = '''    def sanitize_action(self, action_name, game_state):
        """
        Only block actions outside the current task action set.

        Do not rewrite shoot/use based on enemy visibility here.
        The reward system teaches good/bad usage.
        """

        allowed = self.get_allowed_actions()

        if action_name not in allowed:
            return allowed[0] if allowed else "move_forward"

        return action_name

'''

text, n = re.subn(
    r"    def sanitize_action\(self, action_name, game_state\):.*?(?=\n    def _action_name_to_index\(self, action_name\):)",
    sanitize,
    text,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit(f"Expected to replace sanitize_action once, replaced {n}")

# ---------------------------------------------------------
# 8. Add task reward helper
# ---------------------------------------------------------
helper = '''    def compute_curriculum_task_reward(
        self,
        post_game_state,
        action_name,
        distance_moved,
        director_result=None,
        done=False,
        truncated=False,
    ):
        """
        One current task = one clear reward objective.

        This prevents the old all-in-one reward system from teaching loops.
        """

        cfg = self.task_config
        reward = 0.0
        events = []

        x = post_game_state.get("x")
        y = post_game_state.get("y")

        # -----------------------------
        # Navigation / target progress
        # -----------------------------
        if cfg.enable_navigation and x is not None and y is not None:
            tx = float(cfg.target_x)
            ty = float(cfg.target_y)
            dist = ((float(x) - tx) ** 2 + (float(y) - ty) ** 2) ** 0.5

            if not hasattr(self, "task_best_distance") or self.task_best_distance is None:
                self.task_best_distance = dist

            if not hasattr(self, "task_last_distance") or self.task_last_distance is None:
                self.task_last_distance = dist

            improvement = self.task_last_distance - dist

            if improvement > 1.0:
                shaped = min(
                    float(cfg.reward_distance_progress_cap),
                    improvement * float(cfg.reward_distance_progress_scale),
                )
                reward += shaped
                events.append("task_distance_progress")

            if dist < self.task_best_distance - 8.0:
                self.task_best_distance = dist
                reward += float(cfg.reward_new_best_distance)
                events.append("task_new_best_distance")

            if dist <= float(cfg.success_distance):
                reward += float(cfg.reward_task_success)
                events.append("task_target_reached")

            self.task_last_distance = dist

        # -----------------------------
        # Combat
        # -----------------------------
        kill_count = int(post_game_state.get("kill_count", 0) or 0)
        previous_kill_count = int(getattr(self, "task_previous_kill_count", kill_count) or 0)
        kill_delta = max(0, kill_count - previous_kill_count)

        if cfg.enable_combat and kill_delta > 0:
            reward += float(cfg.reward_kill) * kill_delta
            events.append(f"task_kill_x{kill_delta}")

        self.task_previous_kill_count = kill_count

        enemy_visible = bool(post_game_state.get("enemy_visible", False))
        enemy_centered = bool(post_game_state.get("enemy_centered", False))
        ammo = int(post_game_state.get("ammo", 0) or 0)

        if cfg.enable_combat and action_name == "shoot":
            if enemy_visible and enemy_centered and ammo > 0:
                reward += float(cfg.reward_good_shot)
                events.append("task_good_shot")
            elif enemy_visible and ammo > 0:
                reward += float(cfg.penalty_bad_shot)
                events.append("task_poor_aim_shot")
            else:
                reward += float(cfg.penalty_wasted_shot)
                events.append("task_wasted_shot")

        # Combat task success.
        if cfg.enable_combat and cfg.required_kills > 0 and kill_count >= cfg.required_kills:
            reward += float(cfg.reward_task_success)
            events.append("task_required_kills_met")

        # -----------------------------
        # Items/resources
        # -----------------------------
        item_count = int(post_game_state.get("item_count", 0) or 0)
        previous_item_count = int(getattr(self, "task_previous_item_count", item_count) or 0)
        item_delta = max(0, item_count - previous_item_count)

        if cfg.enable_items and item_delta > 0:
            reward += float(cfg.reward_item) * item_delta
            events.append(f"task_item_x{item_delta}")

        self.task_previous_item_count = item_count

        if cfg.enable_items and cfg.required_items > 0 and item_count >= cfg.required_items:
            reward += float(cfg.reward_task_success)
            events.append("task_required_items_met")

        # -----------------------------
        # Use/doors
        # -----------------------------
        if cfg.enable_use and action_name == "use":
            door_visible = bool(post_game_state.get("door_visible", False))
            door_centered = bool(post_game_state.get("door_centered", False))
            near_use_point = bool(post_game_state.get("near_use_point", False))
            opened_door = bool(post_game_state.get("opened_door", False))
            scene_label = post_game_state.get("scene_label")

            useful_use = (
                door_visible
                or door_centered
                or near_use_point
                or opened_door
                or scene_label in ["door_or_button", "switch", "locked_door"]
            )

            if useful_use:
                reward += float(cfg.reward_good_use)
                events.append("task_good_use")
                self.task_successful_uses = getattr(self, "task_successful_uses", 0) + 1
            else:
                reward += float(cfg.penalty_use_spam)
                events.append("task_use_spam")

        if cfg.enable_use and cfg.required_uses > 0:
            if getattr(self, "task_successful_uses", 0) >= cfg.required_uses:
                reward += float(cfg.reward_task_success)
                events.append("task_required_uses_met")

        # -----------------------------
        # Reroute/wall handling
        # -----------------------------
        if cfg.enable_reroute:
            pushing_action = action_name in ["move_forward", "strafe_left", "strafe_right"]
            if pushing_action and float(distance_moved or 0.0) < 1.0:
                self.wall_push_steps = getattr(self, "wall_push_steps", 0) + 1
            else:
                self.wall_push_steps = max(0, getattr(self, "wall_push_steps", 0) - 1)

            if self.wall_push_steps >= 6:
                reward += float(cfg.penalty_wall_push)
                events.append("task_wall_push")

        # -----------------------------
        # Level completion / timeout
        # -----------------------------
        if cfg.enable_exit and done:
            reward += float(cfg.reward_task_success)
            events.append("task_level_done")

        if truncated and not done:
            reward += float(cfg.penalty_timeout)
            events.append("task_timeout")

        return reward, events

'''

if "def compute_curriculum_task_reward" not in text:
    marker = "    def _get_game_vars(self, state):\n"
    if marker not in text:
        raise SystemExit("Could not find _get_game_vars marker")
    text = text.replace(marker, helper + "\n" + marker, 1)

# ---------------------------------------------------------
# 9. Reset task state each episode
# ---------------------------------------------------------
reset_marker = "        self.route_zones_reached = set()\n        self.route_zones_reached = set()\n        self.route_progress_level = 0\n"

reset_insert = """        self.route_zones_reached = set()
        self.route_zones_reached = set()
        self.route_progress_level = 0

        # Reset current small-task curriculum state.
        self.task_best_distance = None
        self.task_last_distance = None
        self.task_previous_kill_count = 0
        self.task_previous_item_count = 0
        self.task_successful_uses = 0
        self.wall_push_steps = 0
"""

if "self.task_best_distance = None" not in text:
    if reset_marker not in text:
        raise SystemExit("Could not find reset route_zones marker")
    text = text.replace(reset_marker, reset_insert, 1)

# ---------------------------------------------------------
# 10. Replace old mixed reward block in step()
# ---------------------------------------------------------
old_block = """        reward += self.spawn_escape_reward(post_game_state)
        reward += self.route_progress_reward(post_game_state)
        reward += self.main_goal_progress_reward(director_result)
        reward += self.goal_progress_reward(director_result)
        reward += self.stagnation_penalty(
            game_state=post_game_state,
            director_result=director_result,
        )
        reward += self.combat_movement_reward(
            game_state=post_game_state,
            action_name=action_name,
            distance_moved=distance_moved,
        )

        obs = self._make_obs()
        reward += self.route_progress_reward(post_game_state)
        reward += self.goal_attraction_bubble_reward(post_game_state)
        self.update_curriculum_from_progress(post_game_state)
        reward += self.exit_distance_progress_reward(post_game_state)
"""

new_block = """        task_reward, task_events = self.compute_curriculum_task_reward(
            post_game_state=post_game_state,
            action_name=action_name,
            distance_moved=distance_moved,
            director_result=director_result,
            done=done,
            truncated=truncated,
        )
        reward += task_reward

        # Optional legacy rewards are controlled by the task config.
        if getattr(self.task_config, "enable_spawn_escape", False):
            reward += self.spawn_escape_reward(post_game_state)

        if getattr(self.task_config, "enable_goal_bubbles", False):
            reward += self.goal_attraction_bubble_reward(post_game_state)

        if getattr(self.task_config, "enable_navigation", False):
            reward += self.stagnation_penalty(
                game_state=post_game_state,
                director_result=director_result,
            )

        obs = self._make_obs()
        self.update_curriculum_from_progress(post_game_state)
"""

if old_block not in text:
    raise SystemExit("Could not find old mixed reward block in step(). Your file may have changed.")
text = text.replace(old_block, new_block, 1)

# ---------------------------------------------------------
# 11. Add task info into info dict
# ---------------------------------------------------------
text = text.replace(
    '"director_result": director_result,',
    '"director_result": director_result,\n            "task_name": self.task_config.name,\n            "task_reward": locals().get("task_reward", 0.0),\n            "task_events": locals().get("task_events", []),',
    1,
)

p.write_text(text)
print("Applied curriculum task system patch.")
