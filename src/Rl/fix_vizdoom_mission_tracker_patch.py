from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_mission_tracker_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add mission tracker imports
# ---------------------------------------------------------

import_line = "from navigation.mission_plan import MissionTracker, get_freedoom1_e1m1_mission\n"

if import_line not in text:
    # Put it near other navigation/reward imports if possible.
    marker = "import numpy as np\n"
    if marker in text:
        text = text.replace(marker, marker + import_line, 1)
    else:
        text = import_line + text

# ---------------------------------------------------------
# 2. Add self.mission_tracker in __init__
# ---------------------------------------------------------

init_insert = '''        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())
        self.enable_doom_brain_reward = True
'''

if "self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())" not in text:
    # Best common marker in most env files.
    markers = [
        "        self.current_step = 0\n",
        "        self.step_count = 0\n",
        "        self.episode_reward = 0.0\n",
    ]

    inserted = False

    for marker in markers:
        if marker in text:
            text = text.replace(marker, marker + init_insert, 1)
            inserted = True
            break

    if not inserted:
        raise SystemExit(
            "Could not find a safe __init__ marker. "
            "Search for 'def __init__' and add mission_tracker manually."
        )

# ---------------------------------------------------------
# 3. Reset mission tracker inside reset()
# ---------------------------------------------------------

reset_insert = '''        if hasattr(self, "mission_tracker"):
            self.mission_tracker.reset()
'''

if "self.mission_tracker.reset()" not in text:
    reset_markers = [
        "        self.current_step = 0\n",
        "        self.step_count = 0\n",
        "        self.episode_reward = 0.0\n",
    ]

    inserted = False

    # Prefer inserting after a reset() occurrence, not the __init__ one.
    reset_pos = text.find("    def reset(")
    if reset_pos != -1:
        before = text[:reset_pos]
        after = text[reset_pos:]

        for marker in reset_markers:
            if marker in after:
                after = after.replace(marker, marker + reset_insert, 1)
                inserted = True
                break

        text = before + after

    if not inserted:
        print("Warning: could not find reset marker. You may need to add mission_tracker.reset() manually.")

p.write_text(text)
print("Patched VizDoomEnv mission_tracker.")
