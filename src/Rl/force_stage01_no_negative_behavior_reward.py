from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_force_stage01_no_negative_behavior_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

guard = '''        # Final Stage 01 safety guard:
        # no negative behavior-judge reward during basic navigation learning.
        task_name_guard = str(getattr(self, "task_name", "") or "").lower()
        task_cfg_guard = str(getattr(getattr(self, "task_config", None), "name", "") or "").lower()
        if reward < 0.0 and (
            "stage_01" in task_name_guard
            or "stage_01" in task_cfg_guard
            or "navigation" in task_name_guard
            or "navigation" in task_cfg_guard
        ):
            reward = 0.0

'''

if "Final Stage 01 safety guard" in text:
    print("Stage 01 final safety guard already exists.")
else:
    # Insert immediately before the behavior_judge_reward debug print.
    pattern = r'(\s*print\(\s*f"\[behavior_judge_reward\].*?\n)'
    match = re.search(pattern, text, flags=re.S)

    if not match:
        # Fallback: insert before return reward inside behavior_judge_reward.
        pattern2 = r'(\s*return reward\s*\n)'
        matches = list(re.finditer(pattern2, text))
        if not matches:
            raise SystemExit("Could not find behavior_judge_reward print or return reward.")
        insert_at = matches[0].start()
        text = text[:insert_at] + guard + text[insert_at:]
        print("Inserted guard before first return reward fallback.")
    else:
        insert_at = match.start()
        text = text[:insert_at] + guard + text[insert_at:]
        print("Inserted guard before behavior_judge_reward print.")

p.write_text(text)
print("Patched forced Stage 01 negative behavior reward suppression.")
