from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_reroute_reset")
backup.write_text(text)
print(f"Backup saved to {backup}")

marker = "        self.step_count = 0\n"

insert = """        self.reroute_stuck_steps = 0
        self.reroute_override_steps = 0
        self.last_reroute_action = None
        self.last_reroute_reason = None
        self.wall_push_steps = 0
        self.no_exit_progress_steps = 0
"""

# Add this only inside reset. Use the second occurrence of self.step_count = 0.
parts = text.split(marker)

if len(parts) < 3:
    raise SystemExit("Expected at least two self.step_count = 0 occurrences: __init__ and reset")

# parts[0] + marker + parts[1] is first occurrence.
# Insert after second occurrence.
text = marker.join(parts[:2]) + marker + insert + marker.join(parts[2:])

p.write_text(text)
print("Added reroute reset state.")
