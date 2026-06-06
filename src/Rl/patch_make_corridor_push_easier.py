from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_make_corridor_push_easier")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            {
                "name": "corridor_push",
                "x": -360.0,
                "y": 470.0,
                "radius": 95.0,
                "min_y": 445.0,
                "max_y": 510.0,
                "reward": 0.50,
            },'''

new = '''            {
                "name": "corridor_push",
                "x": -360.0,
                "y": 445.0,
                "radius": 100.0,
                "min_y": 430.0,
                "max_y": 490.0,
                "reward": 0.50,
            },'''

if old not in text:
    raise SystemExit("Could not find corridor_push block. Paste canonical_route if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Made corridor_push easier/closer after turn_exit.")
