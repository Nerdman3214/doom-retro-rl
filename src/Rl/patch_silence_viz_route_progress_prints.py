from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_silence_viz_route_prints")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Strongly silence any viz_route_progress print block.
text = re.sub(
    r'''[ \t]*print\(
[ \t]*f"\[viz_route_progress\][\s\S]*?\)
''',
    '''            # route bubble debug disabled for exit-first training
            pass
''',
    text,
)

# If any raw string remains, disable by replacing label text.
text = text.replace("[viz_route_progress]", "[route_debug_disabled]")

p.write_text(text)
print("Silenced viz_route_progress prints.")
