from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_safe_shoot_use")
backup.write_text(text)
print(f"Backup saved to {backup}")

flag_block = '''        # Skill test:
        # Let PPO learn movement + shooting + use from the start.
        self.allow_shooting_in_stage0 = True
        self.allow_use_in_stage0 = True
        self.curriculum_movement_plus_shooting = True
        self.curriculum_use_doors_early = True

        # Learnable shoot/use rewards.
        self.wasted_shot_penalty = -12.0
        self.poor_aim_shot_penalty = -6.0
        self.good_shot_reward = 12.0
        self.good_use_reward = 10.0
        self.bad_use_spam_penalty = -2.0
'''

if "self.curriculum_use_doors_early = True" not in text:
    markers = [
        "        self.exit_first_mode = True\n",
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
        "        self.rule_enforcer_reward_scale = 1.0\n",
    ]

    for marker in markers:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            break
    else:
        raise SystemExit("Could not find a safe marker to insert shoot/use flags.")

# Make old harsh values learnable.
text = text.replace("self.wasted_shot_penalty = -60.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.wasted_shot_penalty = -24.0", "self.wasted_shot_penalty = -12.0")
text = text.replace("self.poor_aim_shot_penalty = -24.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.poor_aim_shot_penalty = -12.0", "self.poor_aim_shot_penalty = -6.0")
text = text.replace("self.good_shot_reward = 18.0", "self.good_shot_reward = 12.0")
text = text.replace("self.good_shot_reward = 8.0", "self.good_shot_reward = 12.0")

p.write_text(text)
print("Applied safe shoot/use config patch.")
