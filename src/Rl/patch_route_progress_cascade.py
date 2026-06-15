from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_route_progress_cascade")
backup.write_text(text)
print(f"Backup saved to {backup}")

pattern = r'''        zone = route_zones\[next_idx\]
        if not isinstance\(zone, dict\):
            return 0\.0

        name = zone\.get\("name", f"route_zone_\{next_idx\}"\)
        zx = float\(zone\.get\("x", 0\.0\)\)
        zy = float\(zone\.get\("y", 0\.0\)\)
        radius = float\(zone\.get\("radius", 128\.0\)\)

        dist = \(\(x - zx\) \*\* 2 \+ \(y - zy\) \*\* 2\) \*\* 0\.5

        min_y = zone\.get\("min_y", None\)
        max_y = zone\.get\("max_y", None\)

        if min_y is not None and y < float\(min_y\):
            return 0\.0

        if max_y is not None and y > float\(max_y\):
            return 0\.0

        if dist > radius:
            return 0\.0

        self\.route_zones_reached\.add\(name\)
        self\.route_progress_level = next_idx \+ 1

        zone_reward = float\(forced_route_reward\(name\)\)

        if hasattr\(self, "reward_manager"\) and self\.reward_manager is not None:
            self\.reward_manager\.add\(f"viz_route_progress_\{name\}", zone_reward\)

        print\(
            f"\[viz_route_progress\] reached=\{name!r\} "
            f"level=\{self\.route_progress_level\} "
            f"x=\{x:\.1f\} y=\{y:\.1f\} reward=\{zone_reward:\.2f\}"
        \)

        return zone_reward
'''

replacement = '''        # Cascade route progress:
        # If the agent is inside several overlapping bridge zones on the same
        # frame, reward all reachable next zones instead of waiting one frame.
        total_reward = 0.0
        max_cascade = 4

        for _ in range(max_cascade):
            next_idx = int(getattr(self, "route_progress_level", 0))

            if next_idx >= len(route_zones):
                break

            zone = route_zones[next_idx]
            if not isinstance(zone, dict):
                break

            name = zone.get("name", f"route_zone_{next_idx}")
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            radius = float(zone.get("radius", 128.0))

            dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

            min_y = zone.get("min_y", None)
            max_y = zone.get("max_y", None)

            if min_y is not None and y < float(min_y):
                break

            if max_y is not None and y > float(max_y):
                break

            if dist > radius:
                break

            self.route_zones_reached.add(name)
            self.route_progress_level = next_idx + 1

            zone_reward = float(forced_route_reward(name))
            total_reward += zone_reward

            if hasattr(self, "reward_manager") and self.reward_manager is not None:
                self.reward_manager.add(f"viz_route_progress_{name}", zone_reward)

            print(
                f"[viz_route_progress] reached={name!r} "
                f"level={self.route_progress_level} "
                f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
            )

        return total_reward
'''

new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Could not replace route progress single-zone block. replacements={count}")

p.write_text(new_text)
print("Patched route_progress_reward to cascade nearby route zones.")
