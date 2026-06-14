from pathlib import Path
import re

p = Path("navigation/level_guides.py")
text = p.read_text()

backup = Path("navigation/level_guides.py.backup_before_exit_first")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Turn all route-zone rewards except level_exit to 0.
def zero_non_exit_rewards(match):
    block = match.group(0)
    if '"name": "level_exit"' in block:
        block = re.sub(r'"reward":\s*[-0-9.]+', '"reward": 100.0', block)
        block = re.sub(r'"radius":\s*[-0-9.]+', '"radius": 160.0', block)
        return block
    return re.sub(r'"reward":\s*[-0-9.]+', '"reward": 0.0', block)

text = re.sub(
    r'\{\s*"name":\s*"[^"]+".*?\}',
    zero_non_exit_rewards,
    text,
    flags=re.DOTALL,
)

# Make sure main goal is the exit and has bigger radius for training.
text = re.sub(
    r'"main_goal":\s*\{\s*"name":\s*"level_exit",\s*"x":\s*-400\.0,\s*"y":\s*1296\.0,\s*"radius":\s*[-0-9.]+,\s*\}',
    '"main_goal": {\n            "name": "level_exit",\n            "x": -400.0,\n            "y": 1296.0,\n            "radius": 160.0,\n        }',
    text,
    flags=re.DOTALL,
)

p.write_text(text)
print("Patched level_guides.py for exit-first training.")
