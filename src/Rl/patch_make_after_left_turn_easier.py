from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_make_after_left_turn_easier")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 520.0,
                "radius": 100.0,
                "min_y": 490.0,
                "max_y": 575.0,
                "reward": 0.60,
            },'''

new = '''            {
                "name": "after_left_turn",
                "x": -360.0,
                "y": 460.0,
                "radius": 105.0,
                "min_y": 432.0,
                "max_y": 520.0,
                "reward": 0.60,
            },'''

if old not in text:
    raise SystemExit("Could not find after_left_turn block. Paste canonical_route if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Made after_left_turn easier/closer after corridor_push.")
