from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_make_post_corridor_easier")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            {
                "name": "post_corridor",
                "x": -345.0,
                "y": 510.0,
                "radius": 120.0,
                "min_y": 470.0,
                "max_y": 585.0,
                "reward": 0.85,
            },'''

new = '''            {
                "name": "post_corridor",
                "x": -345.0,
                "y": 445.0,
                "radius": 125.0,
                "min_y": 430.0,
                "max_y": 520.0,
                "reward": 0.85,
            },'''

if old not in text:
    raise SystemExit("Could not find post_corridor block. Paste canonical_route if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Moved post_corridor down near the current post_turn_push bottleneck.")
