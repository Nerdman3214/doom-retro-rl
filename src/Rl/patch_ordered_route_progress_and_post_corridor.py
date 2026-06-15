from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_ordered_route_progress")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Make post_corridor farther and tighter.
# Old radius=170 was too generous and fired near y=390.
# ------------------------------------------------------------
text = text.replace(
'''        post_zone = {
            "name": "post_corridor",
            "x": -340.0,
            "y": 560.0,
            "radius": 170.0,
            "reward": 0.60,
        }
''',
'''        post_zone = {
            "name": "post_corridor",
            "x": -340.0,
            "y": 560.0,
            "radius": 90.0,
            "reward": 0.60,
        }
'''
)

# ------------------------------------------------------------
# 2. Replace route_progress_reward with an ordered version.
# This prevents post_corridor from triggering before the route reaches it.
# ------------------------------------------------------------
new_func = '''    def route_progress_reward(self, game_state):
        self.ensure_post_corridor_route_zone()

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        route_zones = self.level_guide.get("route_zones", [])
        if not isinstance(route_zones, list) or not route_zones:
            return 0.0

        def forced_route_reward(raw_name):
            name = str(raw_name or "").strip().lower()
            name = name.replace("-", "_").replace(" ", "_")

            exact = {
                "spawn_exit_lane": 0.10,
                "sloped_corridor_entry": 0.20,
                "sloped_corridor_mid": 0.30,
                "corridor_left_turn": 0.40,
                "post_corridor": 0.60,
                "exit_approach": 0.80,
                "level_exit": 1.00,
            }

            if name in exact:
                return exact[name]

            if "spawn" in name and "exit" in name:
                return 0.10
            if "sloped" in name and "entry" in name:
                return 0.20
            if "sloped" in name and "mid" in name:
                return 0.30
            if "corridor" in name and ("left" in name or "turn" in name):
                return 0.40
            if "post" in name and "corridor" in name:
                return 0.60
            if "exit" in name and "approach" in name:
                return 0.80
            if "level" in name and "exit" in name:
                return 1.00

            return 0.25

        # Ordered route progress:
        # Only reward the next unreached route zone instead of scanning all zones.
        next_idx = int(getattr(self, "route_progress_level", 0))

        if next_idx < 0:
            next_idx = 0

        if next_idx >= len(route_zones):
            return 0.0

        zone = route_zones[next_idx]
        if not isinstance(zone, dict):
            return 0.0

        name = zone.get("name", f"route_zone_{next_idx}")
        zx = float(zone.get("x", 0.0))
        zy = float(zone.get("y", 0.0))
        radius = float(zone.get("radius", 128.0))

        dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

        if dist > radius:
            return 0.0

        self.route_zones_reached.add(name)
        self.route_progress_level = next_idx + 1

        zone_reward = float(forced_route_reward(name))

        if hasattr(self, "reward_manager") and self.reward_manager is not None:
            self.reward_manager.add(f"viz_route_progress_{name}", zone_reward)

        print(
            f"[viz_route_progress] reached={name!r} "
            f"level={self.route_progress_level} "
            f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
        )

        return zone_reward

'''

pattern = r"    def route_progress_reward\(self, game_state\):\n.*?(?=\n    def goal_heading_reward\(self, game_state\):)"

new_text, count = re.subn(pattern, new_func, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace route_progress_reward. replacements={count}")

p.write_text(new_text)
print("Patched ordered route progress and tightened post_corridor.")
