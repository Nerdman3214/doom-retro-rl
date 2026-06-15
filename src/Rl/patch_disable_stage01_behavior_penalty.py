from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_disable_stage01_behavior_penalty")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''        if bad_prob >= self.behavior_judge_bad_threshold:
            reward = self.behavior_judge_bad_penalty
        elif good_prob >= self.behavior_judge_good_threshold:
            reward = self.behavior_judge_good_reward
'''

new = '''        task_name = str(getattr(self, "task_name", "") or "").lower()
        task_config_name = str(getattr(getattr(self, "task_config", None), "name", "") or "").lower()

        is_stage01_navigation = (
            "stage_01" in task_name
            or "stage_01" in task_config_name
            or task_name == "stage_01_navigation"
            or task_config_name == "stage_01_navigation"
        )

        if bad_prob >= self.behavior_judge_bad_threshold:
            # Stage 01 is basic route learning. Negative RLHF is too harsh here
            # and can collapse the reward before the route is learned.
            if is_stage01_navigation:
                reward = 0.0
            else:
                reward = self.behavior_judge_bad_penalty
        elif good_prob >= self.behavior_judge_good_threshold:
            reward = self.behavior_judge_good_reward
'''

if old not in text:
    print("Exact old behavior-judge block not found.")
    print("Searching for fallback pattern...")

    old2 = '''        if bad_prob >= self.behavior_judge_bad_threshold:
            reward = self.behavior_judge_bad_penalty
'''
    if old2 not in text:
        raise SystemExit("Could not find behavior judge bad penalty block. Paste behavior_judge_reward function.")
    text = text.replace(
        old2,
        '''        task_name = str(getattr(self, "task_name", "") or "").lower()
        task_config_name = str(getattr(getattr(self, "task_config", None), "name", "") or "").lower()
        is_stage01_navigation = (
            "stage_01" in task_name
            or "stage_01" in task_config_name
            or task_name == "stage_01_navigation"
            or task_config_name == "stage_01_navigation"
        )

        if bad_prob >= self.behavior_judge_bad_threshold:
            if is_stage01_navigation:
                reward = 0.0
            else:
                reward = self.behavior_judge_bad_penalty
''',
        1,
    )
else:
    text = text.replace(old, new, 1)

p.write_text(text)
print("Patched Stage 01 behavior-judge penalty suppression.")
