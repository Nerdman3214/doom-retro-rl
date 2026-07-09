from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_route_y_gates")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Replace canonical route with stricter y-gated zones.
# ------------------------------------------------------------
old_route_pattern = r'''        canonical_route = \[
.*?
        \]
'''

new_route = '''        canonical_route = [
            {
                "name": "spawn_exit_lane",
                "x": -416.0,
                "y": 256.0,
                "radius": 55.0,
                "min_y": 240.0,
                "max_y": 285.0,
                "reward": 0.10,
            },
            {
                "name": "sloped_corridor_entry",
                "x": -400.0,
                "y": 285.0,
                "radius": 65.0,
                "min_y": 265.0,
                "max_y": 335.0,
                "reward": 0.20,
            },
            {
                "name": "sloped_corridor_mid",
                "x": -340.0,
                "y": 380.0,
                "radius": 70.0,
                "min_y": 340.0,
                "max_y": 430.0,
                "reward": 0.30,
            },
            {
                "name": "corridor_left_turn",
                "x": -360.0,
                "y": 420.0,
                "radius": 65.0,
                "min_y": 390.0,
                "max_y": 470.0,
                "reward": 0.40,
            },
            {
                "name": "post_corridor",
                "x": -340.0,
                "y": 560.0,
                "radius": 75.0,
                "min_y": 510.0,
                "max_y": 640.0,
                "reward": 0.60,
            },
        ]
'''

new_text, count = re.subn(
    old_route_pattern,
    new_route,
    text,
    count=1,
    flags=re.S,
)

if count != 1:
    raise SystemExit(f"Failed to replace canonical_route. replacements={count}")

text = new_text

# ------------------------------------------------------------
# 2. Add y-gate check inside route_progress_reward after dist calculation.
# ------------------------------------------------------------
old = '''        dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

        if dist > radius:
            return 0.0
'''

new = '''        dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

        min_y = zone.get("min_y", None)
        max_y = zone.get("max_y", None)

        if min_y is not None and y < float(min_y):
            return 0.0

        if max_y is not None and y > float(max_y):
            return 0.0

        if dist > radius:
            return 0.0
'''

if old not in text:
    raise SystemExit("Could not find dist/radius check block in route_progress_reward.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Patched route zone y-gates.")
