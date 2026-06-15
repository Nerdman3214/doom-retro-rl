from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_replace_route_progress_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

new_func = '''    def route_progress_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        reward = 0.0
        route_zones = self.level_guide.get("route_zones", [])

        def forced_route_reward(raw_name):
            name = str(raw_name or "").strip().lower()
            name = name.replace("-", "_").replace(" ", "_")

            # Exact known E1M1 route milestones.
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

            # Fallback keyword matching for duplicate/variant names.
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

        for zone in route_zones:
            name = zone.get("name")
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            radius = float(zone.get("radius", 128.0))

            if name in self.route_zones_reached:
                continue

            dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

            if dist <= radius:
                self.route_zones_reached.add(name)
                self.route_progress_level += 1

                zone_reward = float(forced_route_reward(name))
                reward += zone_reward

                if hasattr(self, "reward_manager") and self.reward_manager is not None:
                    self.reward_manager.add(f"viz_route_progress_{name}", zone_reward)

                print(
                    f"[viz_route_progress] reached={name!r} "
                    f"level={self.route_progress_level} "
                    f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
                )

        return float(reward)

'''

pattern = r"    def route_progress_reward\(self, game_state\):\n.*?(?=\n    def goal_heading_reward\(self, game_state\):)"

new_text, count = re.subn(pattern, new_func, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace route_progress_reward. replacements={count}")

p.write_text(new_text)
print("Replaced route_progress_reward with forced route reward version.")
