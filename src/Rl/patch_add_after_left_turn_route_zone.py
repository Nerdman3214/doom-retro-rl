from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_after_left_turn_zone")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add reward mapping if missing.
text = text.replace(
'''                "corridor_left_turn": 0.40,
                "post_corridor": 0.60,''',
'''                "corridor_left_turn": 0.40,
                "after_left_turn": 0.50,
                "post_corridor": 0.70,''',
1
)

# Add keyword reward fallback if missing.
if 'if "after" in name and "left" in name:' not in text:
    text = text.replace(
'''            if "post" in name and "corridor" in name:
                return 0.60''',
'''            if "after" in name and "left" in name:
                return 0.50
            if "post" in name and "corridor" in name:
                return 0.70''',
1
)

# Replace canonical route with a slightly smoother staircase.
pattern = r'''        canonical_route = \[
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
                "radius": 70.0,
                "min_y": 265.0,
                "max_y": 335.0,
                "reward": 0.20,
            },
            {
                "name": "sloped_corridor_mid",
                "x": -340.0,
                "y": 380.0,
                "radius": 75.0,
                "min_y": 340.0,
                "max_y": 430.0,
                "reward": 0.30,
            },
            {
                "name": "corridor_left_turn",
                "x": -360.0,
                "y": 420.0,
                "radius": 70.0,
                "min_y": 390.0,
                "max_y": 470.0,
                "reward": 0.40,
            },
            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 490.0,
                "radius": 80.0,
                "min_y": 455.0,
                "max_y": 545.0,
                "reward": 0.50,
            },
            {
                "name": "post_corridor",
                "x": -340.0,
                "y": 585.0,
                "radius": 90.0,
                "min_y": 525.0,
                "max_y": 675.0,
                "reward": 0.70,
            },
        ]
'''

new_text, count = re.subn(pattern, new_route, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace canonical_route. replacements={count}")

p.write_text(new_text)
print("Added after_left_turn route zone before post_corridor.")
