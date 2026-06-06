from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_post_turn_push_zone")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add reward mapping.
text = text.replace(
'''                "after_left_turn": 0.60,
                "post_corridor": 0.80,''',
'''                "after_left_turn": 0.60,
                "post_turn_push": 0.70,
                "post_corridor": 0.85,''',
1
)

# Add fallback keyword reward.
if 'if "post_turn_push" in name:' not in text:
    text = text.replace(
'''            if "after" in name and "left" in name:
                return 0.60
            if "post" in name and "corridor" in name:
                return 0.80''',
'''            if "after" in name and "left" in name:
                return 0.60
            if "post_turn_push" in name:
                return 0.70
            if "post" in name and "corridor" in name:
                return 0.85''',
1
)

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
                "radius": 75.0,
                "min_y": 390.0,
                "max_y": 455.0,
                "reward": 0.40,
            },
            {
                "name": "turn_exit",
                "x": -360.0,
                "y": 440.0,
                "radius": 90.0,
                "min_y": 425.0,
                "max_y": 475.0,
                "reward": 0.45,
            },
            {
                "name": "corridor_push",
                "x": -360.0,
                "y": 445.0,
                "radius": 100.0,
                "min_y": 430.0,
                "max_y": 490.0,
                "reward": 0.50,
            },
            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 435.0,
                "radius": 110.0,
                "min_y": 430.0,
                "max_y": 485.0,
                "reward": 0.60,
            },
            {
                "name": "post_turn_push",
                "x": -360.0,
                "y": 455.0,
                "radius": 115.0,
                "min_y": 435.0,
                "max_y": 505.0,
                "reward": 0.70,
            },
            {
                "name": "post_corridor",
                "x": -345.0,
                "y": 510.0,
                "radius": 120.0,
                "min_y": 470.0,
                "max_y": 585.0,
                "reward": 0.85,
            },
        ]
'''

new_text, count = re.subn(pattern, new_route, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace canonical_route. replacements={count}")

p.write_text(new_text)
print("Added post_turn_push bridge before post_corridor.")
