import math
from dataclasses import dataclass, field


@dataclass
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
    Mission model for Freedoom 1 E1M1.

    This does not force secrets before basic survival.
    It tracks navigation, combat, items, doors/use, secrets, then exit.
    """

    return [
        Objective(
            name="spawn_exit_lane",
            kind="route",
            x=-416.0,
            y=256.0,
            radius=96.0,
            reward=0.15,
            required=True,
            hint="leave_spawn",
        ),

        # Sloped corridor path.
        Objective(
            name="sloped_corridor_entry",
            kind="route",
            x=-380.0,
            y=320.0,
            radius=105.0,
            reward=0.30,
            required=True,
            hint="follow_corridor_slope",
        ),
        Objective(
            name="sloped_corridor_mid",
            kind="route",
            x=-300.0,
            y=390.0,
            radius=105.0,
            reward=0.40,
            required=True,
            hint="follow_corridor_slope",
        ),

        # Important: corridor continuation/exit is to the left.
        Objective(
            name="corridor_left_turn",
            kind="route",
            x=-420.0,
            y=430.0,
            radius=125.0,
            reward=0.80,
            required=True,
            hint="turn_left",
        ),

        Objective(
            name="post_corridor_room",
            kind="route",
            x=-430.0,
            y=620.0,
            radius=160.0,
            reward=1.00,
            required=True,
            hint="advance_after_left_turn",
        ),

        # Secrets are optional for now, but tracked for 100%.
        Objective(
            name="secret_sector_52",
            kind="secret",
            x=496.0,
            y=712.0,
            radius=160.0,
            reward=0.60,
            required=False,
            hint="optional_secret",
        ),
        Objective(
            name="secret_sector_86",
            kind="secret",
            x=447.0,
            y=1862.0,
            radius=160.0,
            reward=0.60,
            required=False,
            hint="optional_secret",
        ),
        Objective(
            name="secret_sector_128",
            kind="secret",
            x=1680.0,
            y=-880.0,
            radius=180.0,
            reward=0.60,
            required=False,
            hint="optional_secret",
        ),
        Objective(
            name="secret_sector_132",
            kind="secret",
            x=544.0,
            y=804.0,
            radius=160.0,
            reward=0.60,
            required=False,
            hint="optional_secret",
        ),

        Objective(
            name="exit_approach",
            kind="route",
            x=-400.0,
            y=1000.0,
            radius=200.0,
            reward=1.50,
            required=True,
            hint="advance_to_exit",
        ),
        Objective(
            name="level_exit",
            kind="exit",
            x=-400.0,
            y=1296.0,
            radius=128.0,
            reward=5.00,
            required=True,
            hint="finish_level",
        ),
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
            shaped = min(0.25, improvement * 0.006)
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
                    result["reward"] += min(0.12, exit_improvement * 0.002)
                    self.best_exit_distance = min(self.best_exit_distance, exit_dist)

        result["target"] = self.as_target_dict()
        result["required_index"] = self.required_index
        result["reached"] = sorted(self.reached)
        return result
