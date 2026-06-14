import math
from dataclasses import dataclass, field


@dataclass

class MissionObjective:
    def __init__(
        self,
        name,
        kind,
        x,
        y,
        radius=160.0,
        reward=0.0,
        required=True,
        hint="",
    ):
        self.name = name
        self.kind = kind
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.reward = float(reward)
        self.required = bool(required)
        self.hint = hint

    def as_target_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "hint": self.hint,
        }

class Objective:
    name: str
    kind: str
    x: float
    y: float
    radius: float
    reward: float
    required: bool = True
    hint: str = "advance"


def dist2d(x1, y1, x2, y2):
    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)
    return math.sqrt(dx * dx + dy * dy)


def get_freedoom1_e1m1_mission():
    """
    Exit-only E1M1 mission.

    No corridor/slope/secret objectives for now.
    MissionTracker expects objectives with attributes like .required.
    """

    return [
        MissionObjective(
            name="level_exit",
            kind="exit",
            x=-400.0,
            y=1296.0,
            radius=160.0,
            reward=100.0,
            required=True,
            hint="complete_level",
        )
    ]


class MissionTracker:
    """
    Tracks all objectives, but advances required route objectives sequentially.

    Secrets/items can be rewarded opportunistically.
    Exit is strongly rewarded after route progress.
    """

    def __init__(self, objectives=None):
        self.objectives = objectives or get_freedoom1_e1m1_mission()
        self.reset()

    def reset(self):
        self.reached = set()
        self.required_index = 0
        self.best_target_distance = None
        self.best_exit_distance = None
        self.last_target_name = None

    def required_objectives(self):
        return [obj for obj in self.objectives if obj.required]

    def optional_objectives(self):
        return [obj for obj in self.objectives if not obj.required]

    def current_required(self):
        required = self.required_objectives()
        if self.required_index >= len(required):
            return required[-1]
        return required[self.required_index]

    def current_target(self):
        return self.current_required()

    def as_target_dict(self):
        obj = self.current_target()
        return {
            "name": obj.name,
            "kind": obj.kind,
            "x": obj.x,
            "y": obj.y,
            "radius": obj.radius,
            "hint": obj.hint,
        }

    def update(self, game_state, world_state=None):
        world_state = world_state or {}

        x = game_state.get("x")
        y = game_state.get("y")

        result = {
            "reward": 0.0,
            "events": [],
            "target": self.as_target_dict(),
            "required_index": self.required_index,
            "reached": sorted(self.reached),
        }

        if x is None or y is None:
            return result

        x = float(x)
        y = float(y)

        # Required sequential objective.
        target = self.current_required()
        target_dist = dist2d(x, y, target.x, target.y)

        if self.last_target_name != target.name:
            self.best_target_distance = target_dist
            self.last_target_name = target.name

        if self.best_target_distance is None:
            self.best_target_distance = target_dist

        improvement = self.best_target_distance - target_dist

        # Dense progress toward current objective.
        if improvement > 0.0:
            shaped = min(0.05, improvement * 0.001)
            result["reward"] += shaped
            self.best_target_distance = min(self.best_target_distance, target_dist)
        elif improvement < -24.0:
            result["reward"] -= 0.02

        if target_dist <= target.radius and target.name not in self.reached:
            self.reached.add(target.name)
            result["reward"] += target.reward
            result["events"].append(f"required_reached:{target.name}")
            self.required_index += 1
            self.best_target_distance = None

        # Optional secrets: reward once, but do not force them yet.
        for obj in self.optional_objectives():
            if obj.name in self.reached:
                continue

            od = dist2d(x, y, obj.x, obj.y)
            if od <= obj.radius:
                self.reached.add(obj.name)
                result["reward"] += obj.reward
                result["events"].append(f"optional_reached:{obj.name}")

        # Exit distance always matters, but less than current target.
        exit_obj = None
        for obj in self.objectives:
            if obj.kind == "exit":
                exit_obj = obj
                break

        if exit_obj is not None:
            exit_dist = dist2d(x, y, exit_obj.x, exit_obj.y)

            if self.best_exit_distance is None:
                self.best_exit_distance = exit_dist
            else:
                exit_improvement = self.best_exit_distance - exit_dist
                if exit_improvement > 0:
                    result["reward"] += min(2.00, exit_improvement * 0.020)
                    self.best_exit_distance = min(self.best_exit_distance, exit_dist)

        result["target"] = self.as_target_dict()
        result["required_index"] = self.required_index
        result["reached"] = sorted(self.reached)
        return result
