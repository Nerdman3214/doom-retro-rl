from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_reroute_init")
backup.write_text(text)
print(f"Backup saved to {backup}")

marker = "        self.step_count = 0\n"

insert = """        # -----------------------------------------------------
        # Reroute / stuck recovery tracking
        # -----------------------------------------------------
        # These must be defined in __init__ so linters do not complain
        # about attributes being created later inside step().
        self.reroute_stuck_steps = 0
        self.reroute_override_steps = 0
        self.last_reroute_action = None
        self.last_reroute_reason = None
        self.wall_push_steps = 0
        self.no_exit_progress_steps = 0
"""

if "self.reroute_stuck_steps = 0" not in text:
    if marker not in text:
        raise SystemExit("Could not find self.step_count = 0 marker in __init__")

    text = text.replace(marker, marker + insert, 1)

p.write_text(text)
print("Added reroute attributes to __init__.")
