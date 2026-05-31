import math


class RouteDirector:
    """
    WAD-based route director for Freedoom Phase 1 E1M1.

    Important architecture rule:
    - This class chooses/grades a target.
    - It does not press keys.
    - It returns BOTH new fields and old compatibility fields used by DoomEnv.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_target_name = None
        self.last_distance = None
        self.best_distance = None
        self.no_progress_steps = 0
        self.last_position = None

    def _distance(self, x1, y1, x2, y2):
        dx = float(x1) - float(x2)
        dy = float(y1) - float(y2)
        return math.sqrt(dx * dx + dy * dy)

    def _get_xy(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return None, None

        return float(x), float(y)

    def _choose_main_goal(self, level_guide):
        if not level_guide:
            return None
        return level_guide.get("main_goal")

    def _movement_delta(self, x, y):
        if self.last_position is None:
            self.last_position = (x, y)
            return 0.0

        old_x, old_y = self.last_position
        moved = self._distance(x, y, old_x, old_y)
        self.last_position = (x, y)
        return moved

    def _hint_action_from_vector(self, dx, dy, is_stalled=False):
        """
        This is intentionally weak. The env may log this or use it as a hint,
        but PPO should still control the policy.

        Doom/Freedoom E1M1 real exit is mostly north/up from spawn, so if the
        agent is not progressing, prefer movement over endless turning.
        """

        if is_stalled:
            return "move_backward"

        if abs(dy) >= abs(dx):
            return "move_forward"

        # If the goal is mostly sideways, turning is only a hint.
        return "turn_left" if dx < 0 else "turn_right"

    def _empty_result(self, hint="missing_position"):
        """Return all keys DoomEnv/VizDoomEnv may expect."""
        return {
            "target": None,
            "target_name": None,
            "target_x": None,
            "target_y": None,
            "target_radius": None,
            "objective": None,
            "reason": hint,
            "hint": hint,
            "hint_action": None,
            "dx": 0.0,
            "dy": 0.0,
            "distance": None,
            "distance_delta": 0.0,
            "best_distance": self.best_distance,
            "moved": 0.0,
            "is_stalled": False,
            "no_progress_steps": self.no_progress_steps,
            "reward": 0.0,
            "reward_delta": 0.0,
        }

    def evaluate(self, game_state, action=None, level_guide=None):
        x, y = self._get_xy(game_state)

        if x is None or y is None:
            return self._empty_result("missing_position")

        target = self._choose_main_goal(level_guide)

        if target is None:
            return self._empty_result("missing_goal")

        target_name = target.get("name", "level_exit")
        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))
        radius = float(target.get("radius", 96.0))

        # Direction from player to target.
        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx * dx + dy * dy)

        if self.last_target_name != target_name:
            self.last_target_name = target_name
            self.last_distance = distance
            self.best_distance = distance
            self.no_progress_steps = 0

        if self.last_distance is None:
            distance_delta = 0.0
        else:
            # Positive means closer to the real exit.
            distance_delta = self.last_distance - distance

        moved = self._movement_delta(x, y)

        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance

        if distance_delta > 1.0:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        self.last_distance = distance

        reached = distance <= radius
        is_stalled = moved < 2.0

        if reached:
            hint = "goal_reached"
            reason = "main_goal_reached"
        elif is_stalled or self.no_progress_steps >= 12:
            hint = "recover_unstuck"
            reason = "no_progress"
        elif distance_delta > 2.0:
            hint = "moving_toward_goal"
            reason = "main_goal_progress"
        elif distance_delta < -4.0:
            hint = "moving_away_from_goal"
            reason = "wrong_way"
        else:
            hint = "seek_goal"
            reason = "main_goal"

        hint_action = self._hint_action_from_vector(dx, dy, is_stalled=is_stalled)

        reward_delta = 0.0

        # Keep director shaping small. Env reward handles stronger shaping.
        if distance_delta > 2.0:
            reward_delta += min(0.05, distance_delta / 256.0)
        elif distance_delta < -6.0:
            reward_delta -= min(0.05, abs(distance_delta) / 256.0)

        if is_stalled and self.no_progress_steps >= 8:
            reward_delta -= 0.05

        if reached:
            reward_delta += float(target.get("reward", 10.0))

        return {
            # New/direct fields.
            "target": target_name,
            "target_name": target_name,
            "target_x": tx,
            "target_y": ty,
            "target_radius": radius,
            "distance": distance,
            "distance_delta": distance_delta,
            "best_distance": self.best_distance,
            "moved": moved,
            "is_stalled": is_stalled,
            "no_progress_steps": self.no_progress_steps,
            "hint": hint,
            "reward": reward_delta,

            # Compatibility fields expected by current DoomEnv.
            "dx": dx,
            "dy": dy,
            "hint_action": hint_action,
            "objective": target_name,
            "reason": reason,
            "reward_delta": reward_delta,
        }
    
    def get_next_route_waypoint(self, level_guide, route_progress_level):
        """
        Returns the next route waypoint safely.

        If route data is missing or finished, return None so the normal
        final-exit director can still work.
        """

        if not level_guide:
            return None

        route_zones = level_guide.get("route_zones", [])

        if not route_zones:
            return None

        idx = int(route_progress_level or 0)

        if idx < 0:
            idx = 0

        if idx >= len(route_zones):
            return None

        zone = route_zones[idx]

        if zone.get("x") is None or zone.get("y") is None:
            return None

        return {
            "name": zone.get("name", f"route_zone_{idx}"),
            "x": float(zone["x"]),
            "y": float(zone["y"]),
            "radius": float(zone.get("radius", 128.0)),
            "hint": zone.get("hint", "advance"),
        }
