import math


class RouteDirector:
    """
    Invisible navigation / quest director.

    It does NOT directly control the agent.

    It chooses an objective from the active level guide:
    - main exit
    - nearby secret
    - needed keycard
    - locked door
    - switch/button
    - elevator/door/use point

    The director gives:
    - objective name
    - reason
    - distance
    - small reward for progress
    - optional hint_action such as "use"

    PPO still chooses the real action.
    """

    def __init__(self):
        self.active_target = "explore"
        self.last_distance = None
        self.best_distance = None
        self.last_objective_name = None

    def reset(self):
        self.active_target = "explore"
        self.last_distance = None
        self.best_distance = None
        self.last_objective_name = None

    def find_key_target(self, level_guide, key_name):
        for key_target in level_guide.get("keys", []):
            if key_target.get("key") == key_name:
                return key_target
        return None

    def nearest_target(self, x, y, targets):
        best = None

        for target in targets:
            tx = target.get("x")
            ty = target.get("y")

            if tx is None or ty is None:
                continue

            dx = float(tx) - x
            dy = float(ty) - y
            distance = math.sqrt(dx * dx + dy * dy)

            if best is None or distance < best["distance"]:
                best = {
                    "target": target,
                    "distance": distance,
                    "dx": dx,
                    "dy": dy,
                }

        return best

    def choose_objective(self, game_state, level_guide):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return {
                "name": "explore",
                "type": "explore",
                "reason": "no_position",
                "x": None,
                "y": None,
                "hint_action": None,
                "reward": 0.0,
            }

        x = float(x)
        y = float(y)

        keys_owned = set(game_state.get("keys_owned", []))

        # -----------------------------------------------------
        # 1. If a locked door blocks progress and the key is missing,
        #    switch objective to the needed keycard.
        # -----------------------------------------------------
        for door in level_guide.get("locked_doors", []):
            required_key = door.get("requires_key")

            if required_key and required_key not in keys_owned:
                key_target = self.find_key_target(level_guide, required_key)

                if key_target is not None:
                    return {
                        "name": key_target["name"],
                        "type": "key",
                        "reason": f"need_{required_key}_key_for_{door.get('name')}",
                        "x": key_target["x"],
                        "y": key_target["y"],
                        "hint_action": None,
                        "reward": key_target.get("reward", 6.0),
                        "required_key": required_key,
                    }

        # -----------------------------------------------------
        # 2. If near a known door/elevator/switch/button, suggest use.
        # -----------------------------------------------------
        use_targets = (
            level_guide.get("use_points", [])
            + level_guide.get("switches", [])
            + level_guide.get("locked_doors", [])
        )

        nearest_use = self.nearest_target(x, y, use_targets)

        if nearest_use is not None:
            target = nearest_use["target"]
            radius = float(target.get("radius", 96.0))

            if nearest_use["distance"] <= radius:
                return {
                    "name": target.get("name", "use_point"),
                    "type": target.get("type", "use_point"),
                    "reason": "near_use_point",
                    "x": target.get("x"),
                    "y": target.get("y"),
                    "hint_action": "use",
                    "reward": target.get("reward", 3.0),
                }

        # -----------------------------------------------------
        # 3. Nearby secrets are optional side objectives.
        #    They should not take over the whole route unless close.
        # -----------------------------------------------------
        nearest_secret = self.nearest_target(x, y, level_guide.get("secrets", []))

        if nearest_secret is not None:
            target = nearest_secret["target"]

            if nearest_secret["distance"] <= 160.0:
                return {
                    "name": target.get("name", "secret"),
                    "type": "optional_secret",
                    "reason": "near_secret",
                    "x": target.get("x"),
                    "y": target.get("y"),
                    "hint_action": None,
                    "reward": target.get("reward", 4.0),
                }

        # -----------------------------------------------------
        # 4. Default objective is the map's main goal.
        # -----------------------------------------------------
        main_goal = level_guide.get("main_goal")

        if main_goal is not None:
            return {
                "name": main_goal.get("name", "main_goal"),
                "type": main_goal.get("type", "main_goal"),
                "reason": "main_goal",
                "x": main_goal.get("x"),
                "y": main_goal.get("y"),
                "hint_action": None,
                "reward": main_goal.get("reward", 4.0),
            }

        return {
            "name": "explore",
            "type": "explore",
            "reason": "no_main_goal",
            "x": None,
            "y": None,
            "hint_action": None,
            "reward": 0.0,
        }

    def evaluate(self, game_state, action, level_guide=None):
        if level_guide is None:
            level_guide = {}

        objective = self.choose_objective(game_state, level_guide)

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None or objective.get("x") is None or objective.get("y") is None:
            return {
                "target": objective["name"],
                "objective": objective,
                "reason": objective.get("reason"),
                "distance": None,
                "distance_delta": 0.0,
                "reward_delta": 0.0,
                "hint_action": objective.get("hint_action"),
                "dx": 0.0,
                "dy": 0.0,
            }

        x = float(x)
        y = float(y)

        dx_raw = float(objective["x"]) - x
        dy_raw = float(objective["y"]) - y
        distance = math.sqrt(dx_raw * dx_raw + dy_raw * dy_raw)

        if distance > 0:
            norm_dx = dx_raw / distance
            norm_dy = dy_raw / distance
        else:
            norm_dx = 0.0
            norm_dy = 0.0

        objective_name = objective["name"]

        # Reset distance memory when objective changes.
        if objective_name != self.last_objective_name:
            self.last_distance = None
            self.best_distance = None
            self.last_objective_name = objective_name

        distance_delta = 0.0
        reward_delta = 0.0

        if self.last_distance is not None:
            distance_delta = self.last_distance - distance

            if distance_delta > 2.0:
                reward_delta += 0.25

            elif distance_delta < -6.0:
                reward_delta -= 0.15

        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance
            reward_delta += 0.10

        # If the objective itself says "use", reward use lightly.
        hint_action = objective.get("hint_action")

        if hint_action == "use":
            if action == "use":
                reward_delta += 0.8
            elif action == "move_forward":
                reward_delta -= 0.2

        self.last_distance = distance
        self.active_target = objective_name

        return {
            "target": objective_name,
            "objective": objective,
            "reason": objective.get("reason"),
            "distance": distance,
            "distance_delta": distance_delta,
            "reward_delta": reward_delta,
            "hint_action": hint_action,
            "dx": norm_dx,
            "dy": norm_dy,
        }