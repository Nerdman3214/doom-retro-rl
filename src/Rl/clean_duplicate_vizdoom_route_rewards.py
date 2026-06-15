from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_clean_duplicate_route_rewards")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Disable old inline route progress block.
# This is the block that uses:
#   zone_reward = float(zone.get("reward", 0.05))
# and prints unquoted [viz_route_progress].
# ------------------------------------------------------------
old_block_pattern = r'''
            route_zones = guide\.get\("route_zones", \[\]\)
            next_idx = int\(getattr\(self, "route_progress_level", 0\)\)

            if next_idx < len\(route_zones\):
                zone = route_zones\[next_idx\]
                zx = float\(zone\.get\("x", 0\.0\)\)
                zy = float\(zone\.get\("y", 0\.0\)\)
                radius = float\(zone\.get\("radius", 128\.0\)\)
                zone_reward = float\(zone\.get\("reward", 0\.05\)\)
                name = zone\.get\("name", f"route_zone_\{next_idx\}"\)

                zdx = x - zx
                zdy = y - zy
                zdist = \(zdx \* zdx \+ zdy \* zdy\) \*\* 0\.5

                if zdist <= radius:
                    self\.route_progress_level = next_idx \+ 1
                    self\.route_zones_reached\.add\(name\)
                    reward \+= zone_reward

                    print\(
                        f"\[viz_route_progress\] reached=\{name\} "
                        f"level=\{self\.route_progress_level\} "
                        f"x=\{x:\.1f\} y=\{y:\.1f\} reward=\{zone_reward:\.2f\}"
                    \)
'''

replacement = '''
            # Route progress is handled by route_progress_reward(post_game_state).
            # Old inline route progress block disabled to avoid duplicate zero rewards.
'''

new_text, count = re.subn(
    old_block_pattern,
    replacement,
    text,
    count=1,
    flags=re.X,
)

if count != 1:
    print(f"WARNING: old inline route block replacements={count}")
    print("Trying simpler targeted removal...")

    start_marker = '            route_zones = guide.get("route_zones", [])\n            next_idx = int(getattr(self, "route_progress_level", 0))\n'
    start = text.find(start_marker)

    if start == -1:
        raise SystemExit("Could not find old inline route block start.")

    end_marker = '            # Penalize old east/right drift.\n'
    end = text.find(end_marker, start)

    if end == -1:
        raise SystemExit("Could not find old inline route block end.")

    new_text = text[:start] + replacement + text[end:]
    print("Removed old inline route block using simple marker method.")

# ------------------------------------------------------------
# 2. Remove duplicate route_progress_reward call.
# Keep only the first occurrence.
# ------------------------------------------------------------
call = '        reward += self.route_progress_reward(post_game_state)\n'
first = new_text.find(call)

if first == -1:
    print("WARNING: no route_progress_reward call found.")
else:
    second = new_text.find(call, first + len(call))
    if second != -1:
        new_text = new_text[:second] + '        # duplicate route_progress_reward call removed\n' + new_text[second + len(call):]
        print("Removed duplicate route_progress_reward call.")
    else:
        print("Only one route_progress_reward call found.")

p.write_text(new_text)
print("Cleaned duplicate ViZDoom route rewards.")
