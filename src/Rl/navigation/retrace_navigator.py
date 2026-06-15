from collections import deque


class RetraceNavigator:
    """
    Short-term wall/stuck recovery planner.

    Goal:
    - back away from wall/corner
    - rotate toward the more open side
    - only test forward when the front view is open enough
    """

    def __init__(self, history_size=30):
        self.position_history = deque(maxlen=history_size)
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.total_steps = 0
        self.max_total_steps = 32
        self.preferred_side = "right"
        self.last_escape_action = None

    def reset(self):
        self.position_history.clear()
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.total_steps = 0
        self.preferred_side = "right"
        self.last_escape_action = None

    def update_position(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return

        self.position_history.append((float(x), float(y)))

    def should_start(
        self,
        sensory_state,
        wall_info,
        distance_moved,
        motion,
        stuck_counter,
        wall_contact_steps,
    ):
        situation = sensory_state.get("situation")

        front_ratio = float(wall_info.get("front_ratio", 0.0) or 0.0)
        left_ratio = float(wall_info.get("left_ratio", 0.0) or 0.0)
        right_ratio = float(wall_info.get("right_ratio", 0.0) or 0.0)

        front_blocked = (
            wall_info.get("front_wall", False)
            or front_ratio >= 0.35
        )

        side_wall_pressure = max(left_ratio, right_ratio) >= 0.30

        low_movement = (
            distance_moved is not None
            and distance_moved < 1.0
            and motion is not None
            and motion < 1.5
        )

        repeated_stuck = (
            situation in ["stuck_or_looping", "front_blocked"]
            or stuck_counter >= 4
            or wall_contact_steps >= 2
        )

        return repeated_stuck and (front_blocked or side_wall_pressure or low_movement)

    def start(self, wall_info):
        self.active = True
        self.phase = "back_up"
        self.phase_step = 0
        self.total_steps = 0

        left = float(wall_info.get("left_ratio", 0.0) or 0.0)
        right = float(wall_info.get("right_ratio", 0.0) or 0.0)

        # Lower wall ratio = more open.
        self.preferred_side = "left" if left <= right else "right"

    def stop(self):
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.total_steps = 0
        self.last_escape_action = None

    def front_is_safe(self, wall_info):
        front_ratio = float(wall_info.get("front_ratio", 0.0) or 0.0)
        return front_ratio < 0.30

    def turn_action(self):
        return "turn_left" if self.preferred_side == "left" else "turn_right"

    def opposite_turn_action(self):
        return "turn_right" if self.preferred_side == "left" else "turn_left"

    def strafe_action(self):
        return "strafe_left" if self.preferred_side == "left" else "strafe_right"

    def get_action(
        self,
        sensory_state,
        wall_info,
        distance_moved,
        motion,
        stuck_counter,
        wall_contact_steps,
    ):
        if self.active:
            self.total_steps += 1
            self.phase_step += 1

            if self.total_steps > self.max_total_steps:
                self.stop()
                return None
        else:
            if self.should_start(
                sensory_state=sensory_state,
                wall_info=wall_info,
                distance_moved=distance_moved,
                motion=motion,
                stuck_counter=stuck_counter,
                wall_contact_steps=wall_contact_steps,
            ):
                self.start(wall_info)
            else:
                return None

        if self.phase == "back_up":
            if self.phase_step <= 4:
                self.last_escape_action = "move_backward"
                return "move_backward"

            self.phase = "turn_to_opening"
            self.phase_step = 0

        if self.phase == "turn_to_opening":
            # Stronger rotation, closer to a 90-degree turn.
            if self.phase_step <= 8:
                action = self.turn_action()
                self.last_escape_action = action
                return action

            self.phase = "test_forward"
            self.phase_step = 0

        if self.phase == "test_forward":
            if not self.front_is_safe(wall_info):
                self.phase = "turn_to_opening"
                self.phase_step = 0
                action = self.turn_action()
                self.last_escape_action = action
                return action

            if self.phase_step <= 4:
                self.last_escape_action = "move_forward"
                return "move_forward"

            self.phase = "strafe_search"
            self.phase_step = 0

        if self.phase == "strafe_search":
            if self.phase_step <= 4:
                action = self.strafe_action()
                self.last_escape_action = action
                return action

            self.phase = "opposite_turn"
            self.phase_step = 0

        if self.phase == "opposite_turn":
            if self.phase_step <= 6:
                action = self.opposite_turn_action()
                self.last_escape_action = action
                return action

            self.phase = "test_forward_again"
            self.phase_step = 0

        if self.phase == "test_forward_again":
            if not self.front_is_safe(wall_info):
                self.stop()
                return None

            if self.phase_step <= 4:
                self.last_escape_action = "move_forward"
                return "move_forward"

        self.stop()
        return None