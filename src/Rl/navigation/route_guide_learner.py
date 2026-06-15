import json
from pathlib import Path


class RouteGuideLearner:
    def __init__(self, save_dir="learned_guides"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.events = []

    def record_event(self, level_name, event_type, game_state, extra=None):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return

        tile = (int(float(x) // 64), int(float(y) // 64))

        self.events.append({
            "level": level_name,
            "event_type": event_type,
            "x": float(x),
            "y": float(y),
            "tile": tile,
            "extra": extra or {},
        })

    def save_episode(self, level_name):
        path = self.save_dir / f"{level_name}_events.jsonl"

        with path.open("a") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

        self.events = []