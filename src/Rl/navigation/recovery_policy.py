import math


class HardRecoveryPolicy:
    """
    Shared recovery policy for Doom Retro and ViZDoom.

    Purpose:
    - detect when the agent is not making useful movement
    - detect obstacle / rail / front-blocked pressure
    - commit to a recovery sequence instead of frame-by-frame panic
    - bias back toward the route/exit path

    Recovery sequence:
        1. hard turn / 180 attempt
        2. move backward
        3. strafe away
        4. route gravity toward target
    """

    def __init__(
        self,
        no_movement_threshold_steps=18,
        recovery_steps=28,
        far_wrong_x=700.0,
    ):
        self.no_movement_threshold_steps = int(no_movement_threshold_steps)
        self.recovery_steps_total = int(recovery_steps)
        self.far_wrong_x = float(far_wrong_x)

        self.prev_pos = None
        self.no_movement_steps = 0

        self.active_steps = 0
        self.turn_direction = "left"
        self.reason = None

    def reset(self):
        self.prev_pos = None
        self.no_movement_steps = 0
        self.active_steps = 0
        self.turn_direction = "left"
        self.reason = None

    def update_movement(self, x, y):
        if x is None or y is None:
            return 0.0

        pos = (float(x), float(y))

        if self.prev_pos is None:
            self.prev_pos = pos
            return 0.0

        dx = pos[0] - self.prev_pos[0]
        dy = pos[1] - self.prev_pos[1]
        moved = math.sqrt(dx * dx + dy * dy)

        if moved < 2.0:
            self.no_movement_steps += 1
        else:
            self.no_movement_steps = 0

        self.prev_pos = pos
        return moved

    def should_start(self, game_state, obstacle_info):
        x = game_state.get("x")
        y = game_state.get("y")

        scene_label = game_state.get("scene_label", "")
        sensory_situation = game_state.get("sensory_situation", "")

        front = float(obstacle_info.get("front", 0.0) or 0.0)
        left = float(obstacle_info.get("left", 0.0) or 0.0)
        right = float(obstacle_info.get("right", 0.0) or 0.0)

        obstacle_labels = {
            "front_wall",
            "obstacle",
            "boundary_or_stuck_wall",
            "half_wall_rail",
            "side_rail",
        }

        obstacle_situations = {
            "stuck_or_looping",
            "front_blocked",
            "half_wall_rail",
            "side_rail",
        }

        obstacle_pressure = (
            front >= 0.45
            or left >= 0.55
            or right >= 0.55
            or scene_label in obstacle_labels
            or sensory_situation in obstacle_situations
        )

        no_movement_2sec = (
            self.no_movement_steps >= self.no_movement_threshold_steps
        )

        far_wrong_side = False
        if x is not None:
            try:
                far_wrong_side = float(x) > self.far_wrong_x
            except Exception:
                far_wrong_side = False

        if obstacle_pressure and no_movement_2sec:
            return True, "obstacle_no_movement_2sec"

        if front >= 0.60 and self.no_movement_steps >= 6:
            return True, "front_blocked"

        if far_wrong_side and obstacle_pressure:
            return True, "far_wrong_side_obstacle"

        if sensory_situation == "stuck_or_looping" and self.no_movement_steps >= 8:
            return True, "stuck_or_looping"

        return False, None

    def choose_escape_side(self, game_state, obstacle_info):
        left = float(obstacle_info.get("left", 0.0) or 0.0)
        right = float(obstacle_info.get("right", 0.0) or 0.0)

        x = game_state.get("x")

        try:
            if x is not None and float(x) > self.far_wrong_x:
                return "left"
        except Exception:
            pass

        if left > right + 0.05:
            return "right"

        if right > left + 0.05:
            return "left"

        return self.turn_direction or "left"

    def route_gravity_action(self, game_state, target):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None or not target:
            return "move_forward"

        tx = target.get("x")
        ty = target.get("y")

        if tx is None or ty is None:
            return "move_forward"

        x = float(x)
        tx = float(tx)

        # Freedoom E1M1 problem: far east/right drift.
        if x > 700.0:
            return "turn_left"

        if tx < x - 128.0:
            return "turn_left"

        if tx > x + 128.0:
            return "turn_right"

        return "move_forward"

    def action(self, proposed_action, game_state, obstacle_info, target):
        """
        Return:
            action_name, debug_dict
        """

        self.update_movement(game_state.get("x"), game_state.get("y"))

        should_start, reason = self.should_start(game_state, obstacle_info)

        if should_start and self.active_steps <= 0:
            self.active_steps = self.recovery_steps_total
            self.reason = reason
            self.turn_direction = self.choose_escape_side(game_state, obstacle_info)

        if self.active_steps <= 0:
            return proposed_action, {
                "active": False,
                "reason": None,
                "no_movement_steps": self.no_movement_steps,
            }

        step = self.active_steps
        self.active_steps -= 1

        turn_action = "turn_left" if self.turn_direction == "left" else "turn_right"
        strafe_action = "strafe_left" if self.turn_direction == "left" else "strafe_right"

        # Commit sequence.
        if step > 20:
            chosen = turn_action
        elif step > 14:
            chosen = "move_backward"
        elif step > 8:
            chosen = strafe_action
        else:
            chosen = self.route_gravity_action(game_state, target)

        return chosen, {
            "active": True,
            "reason": self.reason,
            "steps_left": self.active_steps,
            "turn_direction": self.turn_direction,
            "no_movement_steps": self.no_movement_steps,
            "chosen": chosen,
        }
