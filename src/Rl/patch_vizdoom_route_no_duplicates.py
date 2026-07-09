from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_route_no_duplicates")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add reached set in init/reset if missing.
if "self.viz_route_zones_reached = set()" not in text:
    marker = "        self.route_progress_level = 0\n"
    if marker in text:
        text = text.replace(marker, marker + "        self.viz_route_zones_reached = set()\n", 1)

# Add reset clear.
reset_pos = text.find("    def reset(")
if reset_pos != -1 and "self.viz_route_zones_reached.clear()" not in text:
    before = text[:reset_pos]
    after = text[reset_pos:]
    marker = "        self.route_progress_level = 0\n"
    if marker in after:
        after = after.replace(marker, marker + "        self.viz_route_zones_reached.clear()\n", 1)
    text = before + after

# Patch duplicate print/reward behavior in common route-progress function.
# This inserts a duplicate guard before adding reward/printing.
guard = '''            if name in getattr(self, "viz_route_zones_reached", set()):
                return 0.0

            self.viz_route_zones_reached.add(name)
'''

if guard not in text:
    # Look for the line before route progress print.
    marker = '''            print(
                f"[viz_route_progress] reached={name} "
'''
    if marker in text:
        text = text.replace(marker, guard + "\n" + marker, 1)
    else:
        print("Warning: route progress print marker not found. Duplicate guard not inserted.")

p.write_text(text)
print("Patched VizDoom route duplicate guard.")
