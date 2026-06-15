from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_make_post_turn_push_easier")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            {
                "name": "post_turn_push",
                "x": -360.0,
                "y": 455.0,
                "radius": 115.0,
                "min_y": 435.0,
                "max_y": 505.0,
                "reward": 0.70,
            },'''

new = '''            {
                "name": "post_turn_push",
                "x": -360.0,
                "y": 435.0,
                "radius": 120.0,
                "min_y": 430.0,
                "max_y": 490.0,
                "reward": 0.70,
            },'''

if old not in text:
    raise SystemExit("Could not find post_turn_push block. Paste canonical_route if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Moved post_turn_push down to the y≈432 bottleneck.")
