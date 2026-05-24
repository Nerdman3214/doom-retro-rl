from collections import deque


class RetraceNavigator:
    """
    Short-term recovery planner.

    Purpose:
    - prevent repeated wall/object collisions
    - back out of local traps
    - search left/right for openings
    - exit recovery once movement improves

    This does not know the level layout.
    It only uses recent position, movement, wall ratios, and stuck state.
    """

    def __init__(self, history_size=30):
        self.position_history = deque(maxlen=history_size)
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.total_steps = 0
        self.max_total_steps = 24
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

    def should_start(self, sensory_state, wall_info, distance_moved, motion, stuck_counter, wall_contact_steps):
        situation = sensory_state.get("situation")

        front_blocked = (
            wall_info.get("front_wall", False)
            or wall_info.get("front_ratio", 0.0) >= 0.52
        )

        low_movement = (
            distance_moved is not None
            and distance_moved < 1.0
            and motion is not None
            and motion < 1.5
        )

        repeated_stuck = (
            situation in ["stuck_or_looping", "front_blocked"]
            or stuck_counter >= 8
            or wall_contact_steps >= 3
        )

        return repeated_stuck and (front_blocked or low_movement)

    def start(self, wall_info):
        self.active = True
        self.phase = "back_up"
        self.phase_step = 0
        self.total_steps = 0

        left = wall_info.get("left_ratio", 0.0)
        right = wall_info.get("right_ratio", 0.0)

        # Turn toward the more open side.
        if left < right:
            self.preferred_side = "left"
        else:
            self.preferred_side = "right"

    def stop(self):
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.total_steps = 0
        self.last_escape_action = None

    def get_action(self, sensory_state, wall_info, distance_moved, motion, stuck_counter, wall_contact_steps):
        """
        Returns:
            action string or None
        """

        self.total_steps += 1
        self.phase_step += 1

        # Exit retrace mode once movement improves.
        if self.active:
            if distance_moved is not None and motion is not None:
                if distance_moved > 6.0 and motion > 2.0 and self.total_steps >= 4:
                    self.stop()
                    return None

            if self.total_steps > self.max_total_steps:
                self.stop()
                return None

        if not self.active:
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

        # -----------------------------------------------------
        # Recovery plan
        # -----------------------------------------------------

        if self.phase == "back_up":
            if self.phase_step <= 3:
                self.last_escape_action = "move_backward"
                return "move_backward"

            self.phase = "turn_to_opening"
            self.phase_step = 0

        if self.phase == "turn_to_opening":
            if self.phase_step <= 4:
                action = "turn_left" if self.preferred_side == "left" else "turn_right"
                self.last_escape_action = action
                return action

            self.phase = "test_forward"
            self.phase_step = 0

        if self.phase == "test_forward":
            if self.phase_step <= 4:
                self.last_escape_action = "move_forward"
                return "move_forward"

            self.phase = "strafe_search"
            self.phase_step = 0

        if self.phase == "strafe_search":
            if self.phase_step <= 4:
                action = "strafe_left" if self.preferred_side == "left" else "strafe_right"
                self.last_escape_action = action
                return action

            self.phase = "opposite_turn"
            self.phase_step = 0

        if self.phase == "opposite_turn":
            if self.phase_step <= 4:
                action = "turn_right" if self.preferred_side == "left" else "turn_left"
                self.last_escape_action = action
                return action

            self.phase = "test_forward_again"
            self.phase_step = 0

        if self.phase == "test_forward_again":
            if self.phase_step <= 4:
                self.last_escape_action = "move_forward"
                return "move_forward"

            self.stop()
            return None

        return None