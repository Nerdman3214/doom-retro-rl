from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_current_step_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Replace unsafe self.current_step debug gate with a safe step counter.
text = text.replace(
    "if self.current_step % 50 == 0:",
    "if int(getattr(self, 'step_count', getattr(self, 'episode_step', 0))) % 50 == 0:",
)

# Also make sure a step_count exists in __init__ if this env does not already have one.
if "self.step_count" not in text:
    markers = [
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
        "        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())\n",
    ]

    inserted = False
    for marker in markers:
        if marker in text:
            text = text.replace(marker, marker + "        self.step_count = 0\n", 1)
            inserted = True
            break

    if not inserted:
        print("Warning: could not insert self.step_count in __init__.")

# Reset step_count on reset if possible.
if "self.step_count = 0" in text and "self.step_count += 1" not in text:
    # Add increment at the start of step(), after def step line area.
    step_pos = text.find("    def step(")
    if step_pos != -1:
        before = text[:step_pos]
        after = text[step_pos:]

        # Insert after the first line ending with ):
        first_body_marker = "):\n"
        idx = after.find(first_body_marker)
        if idx != -1:
            idx += len(first_body_marker)
            after = after[:idx] + "        self.step_count = int(getattr(self, 'step_count', 0)) + 1\n" + after[idx:]

        text = before + after

p.write_text(text)
print("Fixed rule_enforcer current_step reference.")
