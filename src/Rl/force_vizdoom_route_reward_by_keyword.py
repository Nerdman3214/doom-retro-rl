from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_route_keyword_force")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            if clean_name in forced_rewards:
                zone_reward = float(forced_rewards[clean_name])
            else:
                zone_reward = configured_reward
                if zone_reward <= 0.0:
                    zone_reward = 0.25
'''

new = '''            # Strong normalized matching. Some route zones have duplicate or
            # slightly different names/config values, so exact matching is not enough.
            normalized_name = clean_name.replace("-", "_").replace(" ", "_")

            if "spawn" in normalized_name and "exit" in normalized_name:
                zone_reward = 0.10
            elif "sloped" in normalized_name and "entry" in normalized_name:
                zone_reward = 0.20
            elif "sloped" in normalized_name and ("mid" in normalized_name or "corridor_mid" in normalized_name):
                zone_reward = 0.30
            elif "corridor" in normalized_name and ("left" in normalized_name or "turn" in normalized_name):
                zone_reward = 0.40
            elif "post" in normalized_name and "corridor" in normalized_name:
                zone_reward = 0.60
            elif "exit" in normalized_name and "approach" in normalized_name:
                zone_reward = 0.80
            elif "level" in normalized_name and "exit" in normalized_name:
                zone_reward = 1.00
            elif normalized_name in forced_rewards:
                zone_reward = float(forced_rewards[normalized_name])
            else:
                zone_reward = configured_reward
                if zone_reward <= 0.0:
                    zone_reward = 0.25
'''

if old not in text:
    raise SystemExit("Could not find exact forced reward block. Paste route_progress_reward again.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Forced route rewards by keyword/normalized name.")
