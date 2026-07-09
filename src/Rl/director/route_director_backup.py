import math


class RouteDirector:
    """
    RouteDirector chooses the current navigation target.

    It should not directly control the agent.
    It should not reward random checkpoints too much.
    It should answer:

        Where is the real goal?
        Is the agent getting closer?
        Is the agent stuck?
        What hint should the environment reward system use?

    The env still decides the final reward/action behavior.
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

    def _is_position_stalled(self, x, y):
        """
        Detect if the player is barely moving.

        This does not control movement directly. It only helps the env
        apply anti-stall reward/penalty or recovery logic.
        """

        if self.last_position is None:
            self.last_position = (x, y)
            return False, 0.0

        old_x, old_y = self.last_position
        moved = self._distance(x, y, old_x, old_y)
        self.last_position = (x, y)

        return moved < 2.0, moved

    def evaluate(self, game_state, action=None, level_guide=None):
        x, y = self._get_xy(game_state)

        if x is None or y is None:
            return {
                "target": None,
                "target_name": None,
                "target_x": None,
                "target_y": None,
                "distance": None,
                "distance_delta": 0.0,
                "best_distance": self.best_distance,
                "moved": 0.0,
                "is_stalled": False,
                "no_progress_steps": self.no_progress_steps,
                "hint": "missing_position",
                "reward": 0.0,
            }

        target = self._choose_main_goal(level_guide)

        if target is None:
            return {
                "target": None,
                "target_name": None,
                "target_x": None,
                "target_y": None,
                "distance": None,
                "distance_delta": 0.0,
                "best_distance": self.best_distance,
                "moved": 0.0,
                "is_stalled": False,
                "no_progress_steps": self.no_progress_steps,
                "hint": "missing_goal",
                "reward": 0.0,
            }

        target_name = target.get("name", "level_exit")
        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))
        radius = float(target.get("radius", 96.0))
        dx = tx - x
        dy = ty - y

        distance = self._distance(x, y, tx, ty)

        if self.last_target_name != target_name:
            self.last_target_name = target_name
            self.last_distance = distance
            self.best_distance = distance
            self.no_progress_steps = 0

        if self.last_distance is None:
            distance_delta = 0.0
        else:
            # Positive means closer to goal.
            distance_delta = self.last_distance - distance

        is_stalled, moved = self._is_position_stalled(x, y)

        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance

        if distance_delta > 1.0:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        self.last_distance = distance

        reached = distance <= radius

        if reached:
            hint = "goal_reached"
        elif is_stalled or self.no_progress_steps >= 12:
            hint = "recover_unstuck"
        elif distance_delta > 2.0:
            hint = "moving_toward_goal"
        elif distance_delta < -4.0:
            hint = "moving_away_from_goal"
        else:
            hint = "seek_goal"

        # Keep director reward small. Main reward should be calculated in env.
        reward = 0.0

        if distance_delta > 2.0:
            reward += min(0.05, distance_delta / 256.0)

        if distance_delta < -6.0:
            reward -= min(0.05, abs(distance_delta) / 256.0)

        if is_stalled and self.no_progress_steps >= 8:
            reward -= 0.05

        if reached:
            reward += float(target.get("reward", 10.0))

        return {
            "target": target_name,
            "target_name": target_name,
            "target_x": tx,
            "target_y": ty,
            "target_radius": radius,

            # Compatibility for DoomEnv code that expects these.
            "dx": dx,
            "dy": dy,

            "distance": distance,
            "distance_delta": distance_delta,
            "best_distance": self.best_distance,
            "moved": moved,
            "is_stalled": is_stalled,
            "no_progress_steps": self.no_progress_steps,
            "hint": hint,
            "reward": reward,
        }