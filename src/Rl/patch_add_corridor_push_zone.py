from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_corridor_push_zone")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add reward mapping.
text = text.replace(
'''                "turn_exit": 0.45,
                "after_left_turn": 0.55,
                "post_corridor": 0.75,''',
'''                "turn_exit": 0.45,
                "corridor_push": 0.50,
                "after_left_turn": 0.60,
                "post_corridor": 0.80,''',
1
)

# Add fallback keyword reward.
if 'if "corridor_push" in name:' not in text:
    text = text.replace(
'''            if "turn_exit" in name:
                return 0.45
            if "after" in name and "left" in name:
                return 0.55
            if "post" in name and "corridor" in name:
                return 0.75''',
'''            if "turn_exit" in name:
                return 0.45
            if "corridor_push" in name:
                return 0.50
            if "after" in name and "left" in name:
                return 0.60
            if "post" in name and "corridor" in name:
                return 0.80''',
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
                "y": 470.0,
                "radius": 95.0,
                "min_y": 445.0,
                "max_y": 510.0,
                "reward": 0.50,
            },
            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 520.0,
                "radius": 100.0,
                "min_y": 490.0,
                "max_y": 575.0,
                "reward": 0.60,
            },
            {
                "name": "post_corridor",
                "x": -340.0,
                "y": 610.0,
                "radius": 110.0,
                "min_y": 550.0,
                "max_y": 720.0,
                "reward": 0.80,
            },
        ]
'''

new_text, count = re.subn(pattern, new_route, text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace canonical_route. replacements={count}")

p.write_text(new_text)
print("Added corridor_push bridge zone after turn_exit.")
