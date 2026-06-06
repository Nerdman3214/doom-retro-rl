from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_route_zone_rewards")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# Add route reward table near class-level/config area if missing
# ------------------------------------------------------------
route_reward_block = '''
    ROUTE_ZONE_REWARDS = {
        "spawn_exit_lane": 0.10,
        "sloped_corridor_entry": 0.20,
        "sloped_corridor_mid": 0.30,
        "corridor_left_turn": 0.40,
        "post_corridor": 0.60,
        "exit_approach": 0.80,
        "level_exit": 1.00,
    }

'''

if "ROUTE_ZONE_REWARDS" not in text:
    # Insert after class VizDoomEnv line.
    text = re.sub(
        r"(class\s+VizDoomEnv[^\n]*:\n)",
        r"\1" + route_reward_block,
        text,
        count=1,
    )

# ------------------------------------------------------------
# Replace reward=0.00 route behavior inside route_progress_reward
# ------------------------------------------------------------
# This patch is intentionally defensive: instead of trying to rewrite the whole
# function, it searches for the debug print section and makes sure any detected
# route zone gets reward from ROUTE_ZONE_REWARDS.
#
# Common pattern in your env:
#   [viz_route_progress] reached={zone_name} level={...} x=... y=... reward={reward:.2f}
#
# We inject before that print if not already present.

if "route_zone_reward = float(self.ROUTE_ZONE_REWARDS.get" not in text:
    candidates = [
        'print(f"[viz_route_progress] reached={',
        'print(\n                f"[viz_route_progress] reached=',
        'print(\n            f"[viz_route_progress] reached=',
    ]

    inserted = False

    injection = '''
            # Real route-zone reward.
            # The old detector printed reached zones but often left reward at 0.00.
            route_zone_reward = float(self.ROUTE_ZONE_REWARDS.get(str(zone_name), 0.0))
            reward += route_zone_reward
'''

    for c in candidates:
        idx = text.find(c)
        if idx != -1:
            # Find start of current line.
            line_start = text.rfind("\n", 0, idx) + 1
            text = text[:line_start] + injection + text[line_start:]
            inserted = True
            break

    if not inserted:
        print("WARNING: Could not find viz_route_progress print marker.")
        print("Will add a helper method, but manual inspection may be needed.")

# ------------------------------------------------------------
# If variable is not zone_name in your file, add compatibility aliases.
# ------------------------------------------------------------
# Some versions use reached_zone, zone, name, route_name, or milestone.
# Add a small compatibility block before route_zone_reward if missing.

compat = '''
            # Compatibility with different variable names used in older patches.
            try:
                zone_name
            except NameError:
                zone_name = locals().get("reached_zone", None)
            if zone_name is None:
                zone_name = locals().get("zone", None)
            if zone_name is None:
                zone_name = locals().get("name", None)
            if zone_name is None:
                zone_name = locals().get("route_name", None)
            if zone_name is None:
                zone_name = locals().get("milestone", None)
'''

if "Compatibility with different variable names used in older patches" not in text:
    text = text.replace(
        '''
            # Real route-zone reward.
            # The old detector printed reached zones but often left reward at 0.00.
            route_zone_reward = float(self.ROUTE_ZONE_REWARDS.get(str(zone_name), 0.0))
            reward += route_zone_reward
''',
        compat + '''
            # Real route-zone reward.
            # The old detector printed reached zones but often left reward at 0.00.
            route_zone_reward = float(self.ROUTE_ZONE_REWARDS.get(str(zone_name), 0.0))
            reward += route_zone_reward
''',
        1,
    )

# ------------------------------------------------------------
# Make debug print show the corrected reward if it currently prints reward.
# ------------------------------------------------------------
# Some code prints the local reward variable already. Since we add to reward
# before the print, this should now show nonzero values.

p.write_text(text)
print("Patched route-zone rewards in env/vizdoom_env.py")
