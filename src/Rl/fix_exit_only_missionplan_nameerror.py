from pathlib import Path
import re

p = Path("navigation/mission_plan.py")
text = p.read_text()

backup = Path("navigation/mission_plan.py.backup_before_fix_nameerror")
backup.write_text(text)
print(f"Backup saved to {backup}")

pattern = r'''def get_freedoom1_e1m1_mission\(\):
.*?(?=\ndef |\nclass |\Z)'''

replacement = '''def get_freedoom1_e1m1_mission():
    """
    Exit-only E1M1 mission.

    No corridor/slope/secret objectives.
    The only real objective right now is completing the level.
    """

    return {
        "name": "freedoom1_e1m1_exit_only",
        "objectives": [
            {
                "name": "level_exit",
                "kind": "exit",
                "x": -400.0,
                "y": 1296.0,
                "radius": 160.0,
                "reward": 100.0,
                "required": True,
                "hint": "complete_level",
            }
        ],
        "final_goal": {
            "name": "level_exit",
            "kind": "exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
            "hint": "complete_level",
        },
    }

'''

text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)

if count != 1:
    raise SystemExit(f"Expected to replace get_freedoom1_e1m1_mission once, replaced {count}")

p.write_text(text)
print("Fixed get_freedoom1_e1m1_mission to return plain dict exit-only mission.")
