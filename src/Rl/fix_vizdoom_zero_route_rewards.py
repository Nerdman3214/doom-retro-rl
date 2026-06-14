from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_zero_route_reward_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '            zone_reward = float(zone.get("reward", 0.5))\n'

new = '''            # Some level-guide route zones currently define reward=0.0.
            # That made the detector print reached zones but give no reward.
            configured_reward = float(zone.get("reward", 0.0))

            fallback_rewards = {
                "spawn_exit_lane": 0.10,
                "sloped_corridor_entry": 0.20,
                "sloped_corridor_mid": 0.30,
                "corridor_left_turn": 0.40,
                "post_corridor": 0.60,
                "exit_approach": 0.80,
                "level_exit": 1.00,
            }

            zone_reward = configured_reward
            if zone_reward <= 0.0:
                zone_reward = float(fallback_rewards.get(str(name), 0.25))
'''

if old not in text:
    raise SystemExit("Could not find zone_reward line. Paste route_progress_reward again if needed.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Fixed zero route-zone rewards.")
