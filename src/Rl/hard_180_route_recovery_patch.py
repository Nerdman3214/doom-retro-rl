from pathlib import Path
import re

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_hard_180_route_recovery_patch")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add 180 recovery memory fields in __init__
# ---------------------------------------------------------

init_marker = '''        self.recovery_no_move_steps = 0
'''

init_insert = '''        self.hard_180_recovery_steps = 0
        self.hard_180_turn_direction = "left"
        self.hard_180_reason = None
        self.route_gravity_steps = 0
'''

if init_insert not in text:
    if init_marker in text:
        text = text.replace(init_marker, init_marker + init_insert, 1)
    else:
        fallback_marker = '''        self.recent_loop_steps = 0
'''
        fallback_insert = '''        self.recovery_pulse_steps = 0
        self.recovery_escape_side = "right"
        self.recovery_last_position = None
        self.recovery_no_move_steps = 0
        self.hard_180_recovery_steps = 0
        self.hard_180_turn_direction = "left"
        self.hard_180_reason = None
        self.route_gravity_steps = 0
'''
        if fallback_marker not in text:
            print("Warning: could not find recovery init marker.")
        else:
            text = text.replace(fallback_marker, fallback_marker + fallback_insert, 1)

# ---------------------------------------------------------
# 2. Add helper methods before reset()
# ---------------------------------------------------------

helpers = r'''
    def get_current_route_target(self, game_state=None):
        """
        Pick the current route target.

        Prefer the new mission tracker target if it exists.
        Otherwise fall back to the active route bubble or main goal.
        """

        game_state = game_state or {}

        target = game_state.get("director_target")
        if isinstance(target, dict) and target.get("x") is not None and target.get("y") is not None:
            return target

        try:
            if hasattr(self, "mission_tracker"):
                world_state = {"mode": game_state.get("world_mode", "navigate")}
                mission_update = self.mission_tracker.update(game_state, world_state)
                target = mission_update.get("target")
                if isinstance(target, dict) and target.get("x") is not None and target.get("y") is not None:
                    return target
        except Exception:
            pass

        try:
            target = self.get_active_goal_bubble_target(game_state)
            if isinstance(target, dict) and target.get("x") is not None and target.get("y") is not None:
                return target
        except Exception:
            pass

        guide = getattr(self, "level_guide", {}) or {}
        return guide.get("main_goal")


    def route_gravity_action(self, action, game_state, wall_info=None):
        """
        Convert current position + target position into a simple steering action.

        This is not a full pathfinder. It gives the recovery system a bias:
        if the agent is far east/right of the path, rotate/strafe west/left.
        """

        wall_info = wall_info or {}

        x = game_state.get("x")
        y = game_state.get("y")
        angle = game_state.get("angle")

        target = self.get_current_route_target(game_state)

        if x is None or y is None or not target:
            return action

        tx = target.get("x")
        ty = target.get("y")

        if tx is None or ty is None:
            return action

        x = float(x)
        y = float(y)
        tx = float(tx)
        ty = float(ty)

        dx = tx - x
        dy = ty - y

        # Strong map-specific correction:
        # Freedoom E1M1 useful route/exit lane is much farther west than x=700+.
        # If the agent is far right/east, force west/left correction.
        if x > 700:
            front = float(wall_info.get("front_ratio", 0.0) or 0.0)

            if front >= 0.55:
                return "turn_left"

            # Mix turn-left and strafe-left so it actually translates.
            phase = self._step_count % 6
            if phase in [0, 1, 2]:
                return "turn_left"
            if phase in [3, 4]:
                return "strafe_left"
            return "move_forward"

        # If it is moderately off to the right, bias left but less aggressively.
        if x > 250:
            phase = self._step_count % 5
            if phase in [0, 1]:
                return "turn_left"
            if phase == 2:
                return "strafe_left"
            return action

        # General target steering using dx only when angle is unreliable.
        if dx < -128:
            return "turn_left"

        if dx > 128:
            return "turn_right"

        return action


    def should_start_hard_180_recovery(self, game_state, wall_info=None, sensory_state=None):
        """
        Decide when normal recovery is not enough.

        Use hard 180 when:
        - repeated no-position-change
        - sensory says stuck_or_looping
        - agent is far off-route on the wrong/right/east side
        - front is strongly blocked by rail/wall
        """

        wall_info = wall_info or {}
        sensory_state = sensory_state or {}

        x = game_state.get("x")
        y = game_state.get("y")

        situation = sensory_state.get("situation") or game_state.get("sensory_situation")

        front = float(wall_info.get("front_ratio", 0.0) or 0.0)

        no_move_steps = int(getattr(self, "no_position_change_steps", 0) or 0)
        no_goal_steps = int(getattr(self, "no_distance_progress_steps", 0) or 0)

        far_wrong_side = False
        if x is not None:
            try:
                far_wrong_side = float(x) > 700.0
            except Exception:
                far_wrong_side = False

        stuck = (
            situation == "stuck_or_looping"
            or no_move_steps >= 8
            or no_goal_steps >= 20
            or self.stuck_counter >= 4
        )

        front_blocked = front >= 0.60

        if far_wrong_side and (stuck or front_blocked or no_goal_steps >= 10):
            return True, "far_wrong_side"

        if stuck and front_blocked:
            return True, "stuck_front_blocked"

        if no_move_steps >= 15:
            return True, "no_position_change"

        return False, None


    def hard_180_route_recovery_action(self, action, game_state, wall_info=None, sensory_state=None):
        """
        Hard 180 + route gravity recovery.

        This prevents the classic RL Doom problem:
        rotating forever against walls.

        Recovery sequence:
        - commit to turning around for several frames
        - move/strafe away
        - bias back toward the route target
        """

        wall_info = wall_info or {}
        sensory_state = sensory_state or {}

        should_start, reason = self.should_start_hard_180_recovery(
            game_state=game_state,
            wall_info=wall_info,
            sensory_state=sensory_state,
        )

        if should_start and self.hard_180_recovery_steps <= 0:
            self.hard_180_recovery_steps = 18
            self.hard_180_reason = reason

            left = float(wall_info.get("left_ratio", 0.0) or 0.0)
            right = float(wall_info.get("right_ratio", 0.0) or 0.0)

            # If one side is more blocked, turn away from it.
            # If far east/right of route, prefer left turn to rejoin the path.
            x = game_state.get("x")
            try:
                far_east = x is not None and float(x) > 700.0
            except Exception:
                far_east = False

            if far_east:
                self.hard_180_turn_direction = "left"
            elif left > right:
                self.hard_180_turn_direction = "right"
            else:
                self.hard_180_turn_direction = "left"

            print(
                f"[hard_180_start] reason={reason} "
                f"turn={self.hard_180_turn_direction} "
                f"x={game_state.get('x')} y={game_state.get('y')}"
            )

        if self.hard_180_recovery_steps <= 0:
            return action

        step = self.hard_180_recovery_steps
        self.hard_180_recovery_steps -= 1

        turn_action = "turn_left" if self.hard_180_turn_direction == "left" else "turn_right"
        strafe_action = "strafe_left" if self.hard_180_turn_direction == "left" else "strafe_right"

        # Phase 1: commit to rotating. Do not alternate directions.
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

if "def hard_180_route_recovery_action(" not in text:
    marker = "    def reset("
    if marker not in text:
        raise SystemExit("Could not find reset() marker.")
    text = text.replace(marker, helpers + "\n" + marker, 1)
else:
    print("hard_180_route_recovery_action already exists. Skipping helper insert.")

# ---------------------------------------------------------
# 3. Insert pre-action hard 180 recovery after directional recovery
# ---------------------------------------------------------

needle = r'''        if action != before_directional_recovery:
            print(
                f"[override] directional_recovery: "
                f"{before_directional_recovery} -> {action} "
                f"scene={pre_scene_label} "
                f"L={pre_wall_info.get('left_ratio', 0.0):.2f} "
                f"F={pre_wall_info.get('front_ratio', 0.0):.2f} "
                f"R={pre_wall_info.get('right_ratio', 0.0):.2f}"
            )

'''

insert = r'''        # -----------------------------------------------------
        # Hard 180 + route gravity recovery
        # -----------------------------------------------------
        # This wins after normal directional recovery when the agent is
        # repeatedly stuck or far off the useful route.
        before_hard_180 = action

        action = self.hard_180_route_recovery_action(
            action=action,
            game_state=pre_game_state,
            wall_info=pre_wall_info,
            sensory_state=pre_sensory_state_for_recovery,
        )

        if action != before_hard_180:
            print(
                f"[override] hard_180_route_recovery: "
                f"{before_hard_180} -> {action} "
                f"reason={self.hard_180_reason} "
                f"steps_left={self.hard_180_recovery_steps} "
                f"x={pre_game_state.get('x')} "
                f"y={pre_game_state.get('y')}"
            )

'''

if insert not in text:
    if needle not in text:
        print("Warning: directional recovery print block not found. Trying fallback insert after before_directional_recovery block.")
        fallback = '''        if action != before_directional_recovery:
'''
        # Don't do unsafe fallback replacement here.
    else:
        text = text.replace(needle, needle + insert, 1)

# ---------------------------------------------------------
# 4. Prevent wall_bubble from undoing hard 180 while active
# ---------------------------------------------------------

old = '''        if level == "green":
            return action
'''

new = '''        if getattr(self, "hard_180_recovery_steps", 0) > 0:
            return action

        if level == "green":
            return action
'''

if new not in text and old in text:
    text = text.replace(old, new, 1)

# ---------------------------------------------------------
# 5. Make route gravity reward far-east correction
# ---------------------------------------------------------

reward_marker = '''        reward += self.exit_distance_progress_reward(game_state)
'''

reward_insert = '''        # Route gravity correction:
        # The useful path/exit lane is west of the far-right drift area.
        try:
            gx = float(game_state.get("x", 0.0) or 0.0)
            if gx > 700.0:
                if action in ["turn_left", "strafe_left", "move_backward"]:
                    reward += self.add_positive("far_east_route_recovery_action", 5.0)
                elif action in ["move_forward", "turn_right", "strafe_right"]:
                    reward += self.add_penalty("far_east_wrong_direction", -15.0)
        except Exception:
            pass

'''

if reward_insert not in text:
    if reward_marker in text:
        text = text.replace(reward_marker, reward_insert + "\n" + reward_marker, 1)
    else:
        print("Warning: reward marker not found, route gravity reward not inserted.")

p.write_text(text)
print("Applied hard 180 + route gravity recovery patch.")
