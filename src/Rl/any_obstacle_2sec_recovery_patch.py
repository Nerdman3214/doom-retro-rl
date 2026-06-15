from pathlib import Path
import re

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_any_obstacle_2sec_recovery")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add/replace config values in __init__
# ---------------------------------------------------------

marker = '''        self.route_gravity_steps = 0
'''

insert = '''        # Any obstacle/no-movement recovery settings.
        # About 2 seconds. With 0.10-0.12s actions plus overhead, 18 steps is a good test.
        self.no_movement_recovery_threshold_steps = 18
        self.any_obstacle_recovery_steps = 0
        self.last_meaningful_position = None
'''

if insert not in text:
    if marker in text:
        text = text.replace(marker, marker + insert, 1)
    else:
        fallback = '''        self.recovery_no_move_steps = 0
'''
        if fallback in text:
            text = text.replace(fallback, fallback + insert, 1)
        else:
            print("Warning: could not find recovery init marker. Skipping init insert.")

# ---------------------------------------------------------
# 2. Replace should_start_hard_180_recovery with broader any-obstacle logic
# ---------------------------------------------------------

pattern = r'''    def should_start_hard_180_recovery\(self, game_state, wall_info=None, sensory_state=None\):
.*?
        return False, None
'''

replacement = r'''    def should_start_hard_180_recovery(self, game_state, wall_info=None, sensory_state=None):
        """
        Decide when normal recovery is not enough.

        New rule:
        - not just wall
        - ANY obstacle/rail/boundary/front blocker can trigger recovery
        - no useful translation for about 2 seconds can trigger recovery
        """

        wall_info = wall_info or {}
        sensory_state = sensory_state or {}

        x = game_state.get("x")
        y = game_state.get("y")

        situation = (
            sensory_state.get("situation")
            or game_state.get("sensory_situation")
            or "normal_navigation"
        )

        scene_label = (
            game_state.get("scene_label")
            or game_state.get("pre_scene_label")
            or ""
        )

        front = float(wall_info.get("front_ratio", 0.0) or 0.0)
        left = float(wall_info.get("left_ratio", 0.0) or 0.0)
        right = float(wall_info.get("right_ratio", 0.0) or 0.0)

        no_move_steps = int(getattr(self, "no_position_change_steps", 0) or 0)
        no_goal_steps = int(getattr(self, "no_distance_progress_steps", 0) or 0)

        threshold = int(
            getattr(self, "no_movement_recovery_threshold_steps", 18)
        )

        obstacle_labels = {
            "front_wall",
            "obstacle",
            "boundary_or_stuck_wall",
            "half_wall_rail",
            "side_rail",
            "door_or_button",
        }

        obstacle_situations = {
            "stuck_or_looping",
            "front_blocked",
            "spawn_wall_zone",
            "secret_side_area",
            "right_route_area",
            "half_wall_rail",
            "side_rail",
        }

        any_obstacle_label = scene_label in obstacle_labels
        any_obstacle_situation = situation in obstacle_situations

        # Visual/body obstacle pressure. This catches rails, half-walls,
        # corners, monster-blocking-looking barriers, and invisible boundaries.
        obstacle_pressure = (
            front >= 0.45
            or left >= 0.55
            or right >= 0.55
            or any_obstacle_label
            or any_obstacle_situation
        )

        front_or_rail_blocked = (
            front >= 0.60
            or scene_label in {"half_wall_rail", "front_wall", "obstacle", "boundary_or_stuck_wall"}
            or situation == "front_blocked"
        )

        no_movement_for_2sec = no_move_steps >= threshold

        no_goal_progress_for_2sec = no_goal_steps >= threshold

        far_wrong_side = False
        if x is not None:
            try:
                far_wrong_side = float(x) > 700.0
            except Exception:
                far_wrong_side = False

        # Main new rule:
        # Any obstacle + no movement for about 2 seconds = hard recovery.
        if obstacle_pressure and no_movement_for_2sec:
            return True, "any_obstacle_no_movement_2sec"

        # If the agent is not making route progress for about 2 seconds
        # while obstacle pressure exists, recover.
        if obstacle_pressure and no_goal_progress_for_2sec:
            return True, "any_obstacle_no_goal_progress_2sec"

        # Front/rail block should trigger faster because move_forward will
        # usually keep colliding.
        if front_or_rail_blocked and no_move_steps >= 6:
            return True, "front_or_rail_blocked"

        # Far east/right side is still bad on this map.
        if far_wrong_side and (obstacle_pressure or no_goal_steps >= 10):
            return True, "far_wrong_side"

        # Classic stuck fallback.
        if situation == "stuck_or_looping" and no_move_steps >= 8:
            return True, "stuck_loop_no_movement"

        return False, None
'''

text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)

if count == 0:
    print("Warning: could not replace should_start_hard_180_recovery. Function may not exist yet.")
else:
    print(f"Replaced should_start_hard_180_recovery. count={count}")

# ---------------------------------------------------------
# 3. Make hard 180 sequence stronger and more committed
# ---------------------------------------------------------
# Longer turn + movement sequence:
# 24 steps total:
#   8 turn
#   5 backward
#   5 strafe
#   6 route-gravity correction
# ---------------------------------------------------------

text = text.replace(
    "self.hard_180_recovery_steps = 18",
    "self.hard_180_recovery_steps = 24"
)

old_sequence = '''        # Phase 1: commit to rotating. Do not alternate directions.
        if step > 12:
            return turn_action

        # Phase 2: create real translation after turning.
        if step > 8:
            return "move_backward"

        # Phase 3: slide away from wall/rail.
        if step > 4:
            return strafe_action

        # Phase 4: try route gravity, not random forward.
        return self.route_gravity_action(action, game_state, wall_info)
'''

new_sequence = '''        # Phase 1: hard turn / 180 attempt.
        # Do not alternate directions during this phase.
        if step > 16:
            return turn_action

        # Phase 2: create separation behind the agent.
        if step > 11:
            return "move_backward"

        # Phase 3: translate sideways away from obstacle pressure.
        if step > 6:
            return strafe_action

        # Phase 4: gravitate toward the route target.
        return self.route_gravity_action(action, game_state, wall_info)
'''

if old_sequence in text:
    text = text.replace(old_sequence, new_sequence)
else:
    print("Warning: hard 180 sequence block not found. Skipping sequence replacement.")

# ---------------------------------------------------------
# 4. Prevent move_forward during active hard recovery when obstacle pressure is high
# ---------------------------------------------------------

needle = '''        if self.hard_180_recovery_steps <= 0:
            return action
'''

replacement2 = '''        if self.hard_180_recovery_steps <= 0:
            return action

        # During committed hard recovery, do not let PPO/sanitize force
        # move_forward back into an obstacle.
        obstacle_pressure_now = (
            float(wall_info.get("front_ratio", 0.0) or 0.0) >= 0.45
            or sensory_state.get("situation") in {
                "stuck_or_looping",
                "front_blocked",
                "half_wall_rail",
                "side_rail",
            }
            or game_state.get("scene_label") in {
                "front_wall",
                "obstacle",
                "boundary_or_stuck_wall",
                "half_wall_rail",
                "side_rail",
            }
        )
'''

if replacement2 not in text and needle in text:
    text = text.replace(needle, replacement2, 1)

# ---------------------------------------------------------
# 5. Add reward/penalty for 2-second obstacle recovery behavior
# ---------------------------------------------------------

reward_marker = '''        reward += self.exit_distance_progress_reward(game_state)
'''

reward_insert = '''        # Any obstacle / 2-second no-movement recovery shaping.
        try:
            obstacle_scene = game_state.get("scene_label") in {
                "front_wall",
                "obstacle",
                "boundary_or_stuck_wall",
                "half_wall_rail",
                "side_rail",
            }
            obstacle_situation = game_state.get("sensory_situation") in {
                "stuck_or_looping",
                "front_blocked",
                "half_wall_rail",
                "side_rail",
            }
            no_move_steps = int(getattr(self, "no_position_change_steps", 0) or 0)
            threshold = int(getattr(self, "no_movement_recovery_threshold_steps", 18))

            if (obstacle_scene or obstacle_situation) and no_move_steps >= threshold:
                if action in ["turn_left", "turn_right", "move_backward", "strafe_left", "strafe_right"]:
                    reward += self.add_positive("any_obstacle_2sec_recovery_action", 5.0)
                elif action == "move_forward":
                    reward += self.add_penalty("any_obstacle_2sec_push_forward", -15.0)
        except Exception:
            pass

'''

if reward_insert not in text:
    if reward_marker in text:
        text = text.replace(reward_marker, reward_insert + "\n" + reward_marker, 1)
    else:
        print("Warning: reward marker not found. Skipping reward insert.")

p.write_text(text)
print("Applied any-obstacle + 2-second hard recovery patch.")
