from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_force_route_reward_table")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            zone_reward = configured_reward
            if zone_reward <= 0.0:
                zone_reward = float(fallback_rewards.get(str(name), 0.25))
'''

new = '''            # Force our known route milestone rewards.
            # Do not trust level-guide reward=0.0 for core route zones.
            clean_name = str(name).strip().lower()

            forced_rewards = {
                "spawn_exit_lane": 0.10,
                "sloped_corridor_entry": 0.20,
                "sloped_corridor_mid": 0.30,
                "corridor_left_turn": 0.40,
                "post_corridor": 0.60,
                "exit_approach": 0.80,
                "level_exit": 1.00,
            }

            if clean_name in forced_rewards:
                zone_reward = float(forced_rewards[clean_name])
            else:
                zone_reward = configured_reward
                if zone_reward <= 0.0:
                    zone_reward = 0.25
'''

if old not in text:
    raise SystemExit("Could not find previous route reward block. Paste route_progress_reward again.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Forced known route-zone rewards.")
