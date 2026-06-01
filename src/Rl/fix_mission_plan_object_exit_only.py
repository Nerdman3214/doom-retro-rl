from pathlib import Path
import re

p = Path("navigation/mission_plan.py")
text = p.read_text()

backup = Path("navigation/mission_plan.py.backup_before_object_exit_only_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add a tiny objective class only if the file does not already have one.
if "class MissionObjective" not in text:
    insert = '''
class MissionObjective:
    def __init__(
        self,
        name,
        kind,
        x,
        y,
        radius=160.0,
        reward=0.0,
        required=True,
        hint="",
    ):
        self.name = name
        self.kind = kind
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.reward = float(reward)
        self.required = bool(required)
        self.hint = hint

    def as_target_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "hint": self.hint,
        }

'''
    # Put after imports.
    first_class = text.find("class ")
    if first_class != -1:
        text = text[:first_class] + insert + text[first_class:]
    else:
        text = insert + text

# Replace get_freedoom1_e1m1_mission with object-list version.
pattern = r'''def get_freedoom1_e1m1_mission\(\):
.*?(?=\ndef |\nclass |\Z)'''

replacement = '''def get_freedoom1_e1m1_mission():
    """
    Exit-only E1M1 mission.

    No corridor/slope/secret objectives for now.
    MissionTracker expects objectives with attributes like .required.
    """

    return [
        MissionObjective(
            name="level_exit",
            kind="exit",
            x=-400.0,
            y=1296.0,
            radius=160.0,
            reward=100.0,
            required=True,
            hint="complete_level",
        )
    ]

'''

text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)

if count != 1:
    raise SystemExit(f"Expected to replace get_freedoom1_e1m1_mission once, replaced {count}")

p.write_text(text)
print("Fixed mission_plan.py with object-style exit-only objective.")
