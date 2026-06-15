from collections import deque


class SensoryModel:
    """
    Code-based sensory model.

    This is not a neural network yet. It combines:
    - vision label/confidence
    - x/y position
    - wall ratios
    - motion
    - distance moved
    - stuck counter
    - route memory

    Output:
    - situation label
    - confidence
    - recommended action
    - reward hints
    """

    def __init__(self, history_size=20):
        self.position_history = deque(maxlen=history_size)
        self.situation_history = deque(maxlen=history_size)

        self.reached_corridor_this_episode = False
        self.reached_secret_this_episode = False
        self.reached_right_route_this_episode = False

        self.last_safe_tile = None
        self.last_progress_tile = None

    def reset(self):
        self.position_history.clear()
        self.situation_history.clear()

        self.reached_corridor_this_episode = False
        self.reached_secret_this_episode = False
        self.reached_right_route_this_episode = False

        self.last_safe_tile = None
        self.last_progress_tile = None

    def tile_from_position(self, x, y, tile_size=64):
        if x is None or y is None:
            return None

        return (int(float(x) // tile_size), int(float(y) // tile_size))

    def is_spawn_wall_zone(self, x, y):
        if x is None or y is None:
            return False

        x = float(x)
        y = float(y)

        return -520.0 <= x <= -430.0 and 70.0 <= y <= 450.0

    def is_secret_side_area(self, x, y):
        """
        This is the side/secret-ish area you were seeing.
        Keep this separate from the corridor.
        """
        if x is None or y is None:
            return False

        x = float(x)
        y = float(y)

        return -260.0 <= x <= -100.0 and 150.0 <= y <= 380.0

    def is_right_route_area(self, x, y):
        """
        Right-side route from spawn. Adjust later once coordinates are clearer.
        """
        if x is None or y is None:
            return False

        x = float(x)
        y = float(y)

        return 100.0 <= x <= 650.0 and 300.0 <= y <= 470.0

    def repeated_tile_count(self):
        if not self.position_history:
            return 0

        current = self.position_history[-1]
        return sum(1 for tile in self.position_history if tile == current)

    def evaluate(
        self,
        *,
        game_state,
        scene_label,
        scene_confidence,
        wall_info,
        action,
        distance_moved,
        motion,
        stuck_counter,
        wall_contact_steps,
        corridor_reached,
    ):
        x = game_state.get("x")
        y = game_state.get("y")

        tile = self.tile_from_position(x, y)

        if tile is not None:
            self.position_history.append(tile)

        repeated_count = self.repeated_tile_count()

        front_ratio = float(wall_info.get("front_ratio", 0.0))
        left_ratio = float(wall_info.get("left_ratio", 0.0))
        right_ratio = float(wall_info.get("right_ratio", 0.0))

        front_blocked = (
            front_ratio >= 0.55
            or wall_info.get("front_wall", False)
            or (
                scene_label in ["front_wall", "obstacle"]
                and scene_confidence >= 0.75
            )
        )


        side_wall_pressure = (
            left_ratio >= 0.75
            or right_ratio >= 0.75
        )

        movement_failed = (
            action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]
            and distance_moved is not None
            and motion is not None
            and distance_moved < 1.0
            and motion < 1.5
        )

        spawn_wall_zone = self.is_spawn_wall_zone(x, y)
        secret_side_area = self.is_secret_side_area(x, y)
        right_route_area = self.is_right_route_area(x, y)

        door_possible = (
            scene_label == "door_or_button"
            and scene_confidence >= 0.35
        )

        if corridor_reached:
            self.reached_corridor_this_episode = True
            self.last_progress_tile = tile

        if secret_side_area:
            self.reached_secret_this_episode = True
            self.last_progress_tile = tile

        if right_route_area:
            self.reached_right_route_this_episode = True
            self.last_progress_tile = tile

        situation = "normal_navigation"
        confidence = 0.50
        recommended_action = None
        reward_delta = 0.0
        reward_name = None

        if corridor_reached:
            situation = "corridor"
            confidence = 0.95
            reward_delta = 2.0
            reward_name = "sensory_corridor_hold"

            if action == "move_backward":
                reward_delta -= 1.0
                reward_name = "sensory_backing_out_of_corridor"

        elif spawn_wall_zone and self.reached_corridor_this_episode:
            situation = "bad_return_to_spawn_wall"
            confidence = 0.95
            reward_delta = -2.0
            reward_name = "sensory_bad_return_to_spawn_wall"
            recommended_action = "turn_right"

        elif spawn_wall_zone:
            situation = "spawn_wall_zone"
            confidence = 0.90
            reward_delta = -0.8
            reward_name = "sensory_spawn_wall_zone"

            if action == "move_forward":
                recommended_action = "turn_right"

        if door_possible:
            situation = "door_possible"
            confidence = 0.85
            recommended_action = "use"
            reward_delta = 0.8
            reward_name = "sensory_door_possible"

        elif movement_failed or stuck_counter >= 5 or repeated_count >= 8:
            situation = "stuck_or_looping"
            confidence = 0.90
            reward_delta = -1.0
            reward_name = "sensory_stuck_or_looping"

            cycle = len(self.situation_history) % 4
            if cycle == 0:
                recommended_action = "move_backward"
            elif cycle == 1:
                recommended_action = "turn_right"
            elif cycle == 2:
                recommended_action = "strafe_right"
            else:
                recommended_action = "move_forward"

        elif front_blocked:
            situation = "front_blocked"
            confidence = 0.85
            reward_delta = -0.5
            reward_name = "sensory_front_blocked"

            if action == "move_forward":
                if right_ratio < left_ratio:
                    recommended_action = "strafe_right"
                else:
                    recommended_action = "strafe_left"

        elif side_wall_pressure:
            situation = "side_wall_pressure"
            confidence = 0.75
            reward_delta = -0.5
            reward_name = "sensory_side_wall_pressure"

            if right_ratio > left_ratio and action == "strafe_right":
                recommended_action = "strafe_left"

            elif left_ratio > right_ratio and action == "strafe_left":
                recommended_action = "strafe_right"

        elif secret_side_area:
            situation = "secret_side_area"
            confidence = 0.85
            reward_delta = 1.0
            reward_name = "sensory_secret_side_area"

        elif right_route_area:
            situation = "right_route_area"
            confidence = 0.85
            reward_delta = 1.0
            reward_name = "sensory_right_route_area"

        elif scene_label == "open_path" and scene_confidence >= 0.70:
            situation = "open_path"
            confidence = 0.75

            if action in ["move_forward", "strafe_left", "strafe_right"] and distance_moved > 2.0:
                reward_delta = 0.5
                reward_name = "sensory_good_open_movement"

        result = {
            "situation": situation,
            "confidence": confidence,
            "recommended_action": recommended_action,
            "reward_delta": reward_delta,
            "reward_name": reward_name,
            "tile": tile,
            "repeated_tile_count": repeated_count,
            "spawn_wall_zone": spawn_wall_zone,
            "secret_side_area": secret_side_area,
            "right_route_area": right_route_area,
            "front_blocked": front_blocked,
            "side_wall_pressure": side_wall_pressure,
            "movement_failed": movement_failed,
        }

        self.situation_history.append(situation)

        return result
    
    def should_override_action(self, situation, recommended_action, wall_info, stuck_counter, wall_contact_steps):
        if recommended_action is None:
            return False

        if situation in ["normal_navigation", "right_route_area", "secret_side_area"]:
            return False

        front_ratio = float(wall_info.get("front_ratio", 0.0))
        left_ratio = float(wall_info.get("left_ratio", 0.0))
        right_ratio = float(wall_info.get("right_ratio", 0.0))

        real_wall_problem = (
            front_ratio >= 0.55
            or stuck_counter >= 8
            or wall_contact_steps >= 3
            or (front_ratio >= 0.45 and left_ratio >= 0.45 and right_ratio >= 0.45)
        )

        return real_wall_problem