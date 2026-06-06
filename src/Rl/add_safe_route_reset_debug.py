from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_safe_route_reset_debug")
backup.write_text(text)
print(f"Backup saved to {backup}")

needle = '''        self.route_progress_level = 0
        self.route_zones_reached = set()
'''

replacement = '''        self.route_progress_level = 0
        self.route_zones_reached = set()
        print("[route_reset] route progress reset")
'''

if '[route_reset] route progress reset' not in text:
    if needle not in text:
        print("Could not find exact route reset block. Skipping debug print.")
    else:
        text = text.replace(needle, replacement, 1)
        print("Added safe route reset debug print.")
else:
    print("Route reset debug print already exists.")

p.write_text(text)
