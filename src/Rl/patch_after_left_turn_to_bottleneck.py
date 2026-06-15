from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_after_left_turn_bottleneck")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 460.0,
                "radius": 105.0,
                "min_y": 432.0,
                "max_y": 520.0,
                "reward": 0.60,
            },'''

new = '''            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 435.0,
                "radius": 110.0,
                "min_y": 430.0,
                "max_y": 485.0,
                "reward": 0.60,
            },'''

if old not in text:
    raise SystemExit("Could not find after_left_turn block. Paste canonical_route if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Moved after_left_turn to the current y≈432 bottleneck.")
