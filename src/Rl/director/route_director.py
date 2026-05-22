import math


class RouteDirector:
    """
    Invisible navigation director.

    This does not draw anything on the screen.
    It gives the agent hidden guidance toward known useful areas:
    - corridor
    - right-side route
    - secret/use areas

    This is like a lightweight L4D2 / Alien Isolation director for navigation.
    """

    def __init__(self):
        self.targets = {
            "goal": {
                "name": "goal",
                "x": 1200.0,
                "y": -100.0,
                "reward": 4.0,
            },
            "right_route": {
                "name": "right_route",
                "x": 300.0,
                "y": 380.0,
                "reward": 1.0,
            },
            "secret_side": {
                "name": "secret_side",
                "x": -210.0,
                "y": 230.0,
                "reward": 1.0,
            },
        }

        self.active_target = "goal"
        self.last_distance = None
        self.best_distance = None

    def reset(self):
        self.active_target = "right_route"
        self.last_distance = None
        self.best_distance = None

    def choose_target(self, game_state):
        """
        Goal-first director.

        Main target is the real level goal/progression area.
        Right route and secrets are optional helper targets, not the main objective.
        """

        x = game_state.get("x")
        y = game_state.get("y")

        # If your observer exposes a real goal position, use it.
        goal_x = game_state.get("goal_x")
        goal_y = game_state.get("goal_y")

        if goal_x is not None and goal_y is not None:
            self.targets["goal"]["x"] = float(goal_x)
            self.targets["goal"]["y"] = float(goal_y)

        self.active_target = "goal"
        return self.targets["goal"]

    def evaluate(self, game_state, action):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return {
                "target": self.active_target,
                "distance": None,
                "distance_delta": 0.0,
                "reward_delta": 0.0,
                "hint_action": None,
                "dx": 0.0,
                "dy": 0.0,
            }

        x = float(x)
        y = float(y)

        target = self.choose_target(game_state)

        dx = target["x"] - x
        dy = target["y"] - y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0:
            norm_dx = dx / distance
            norm_dy = dy / distance
        else:
            norm_dx = 0.0
            norm_dy = 0.0

        distance_delta = 0.0
        reward_delta = 0.0

        if self.last_distance is not None:
            distance_delta = self.last_distance - distance

            # Moving closer to director target is good.
            if distance_delta > 2.0:
                reward_delta += 0.4

            # Moving away from director target is bad, but not too harsh.
            elif distance_delta < -4.0:
                reward_delta -= 0.3

        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance
            reward_delta += 0.2

        hint_action = None

        # Simple action hint.
        # This is approximate because we do not yet know the player's angle.
        # Later we can use player angle if shared state exposes it.
        if abs(dx) > abs(dy):
            if dx > 0:
                hint_action = "strafe_right"
            else:
                hint_action = "strafe_left"
        else:
            hint_action = "move_forward"

        # Reward following the hint lightly.
        if action == hint_action:
            reward_delta += 0.1

        self.last_distance = distance

        return {
            "target": target["name"],
            "distance": distance,
            "distance_delta": distance_delta,
            "reward_delta": reward_delta,
            "hint_action": hint_action,
            "dx": norm_dx,
            "dy": norm_dy,
        }