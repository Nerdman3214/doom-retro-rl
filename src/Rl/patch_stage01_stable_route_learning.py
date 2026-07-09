from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_stage01_stable_route_learning")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Disable behavior judge penalty during early Stage 01 route learning.
# Keep positive behavior rewards if they happen, but do not punish yet.
# ------------------------------------------------------------
text = text.replace(
'''        if bad_prob >= self.behavior_judge_bad_threshold:
            reward = self.behavior_judge_bad_penalty
        elif good_prob >= self.behavior_judge_good_threshold:
            reward = self.behavior_judge_good_reward
''',
'''        task_name = str(getattr(self, "task_name", "") or "")
        stage_name = str(getattr(getattr(self, "task_config", None), "name", "") or "")

        # Stage 01 is still learning basic route following.
        # Avoid heavy RLHF punishment until route completion is stable.
        if bad_prob >= self.behavior_judge_bad_threshold:
            if "stage_01" in task_name or "stage_01" in stage_name or "navigation" in task_name:
                reward = 0.0
            else:
                reward = self.behavior_judge_bad_penalty
        elif good_prob >= self.behavior_judge_good_threshold:
            reward = self.behavior_judge_good_reward
''',
1
)

# ------------------------------------------------------------
# 2. Make route_progress_reward never reset route level itself.
# Also add a tiny forward-progress shaping reward after corridor_left_turn.
# ------------------------------------------------------------
old = '''        if next_idx >= len(route_zones):
            return 0.0

        zone = route_zones[next_idx]
'''

new = '''        if next_idx >= len(route_zones):
            # Route already complete for this episode.
            # Small shaping reward for staying beyond the corridor turn.
            if y >= 500.0:
                return 0.02
            return 0.0

        # After corridor_left_turn, encourage movement toward post_corridor.
        # This helps the agent discover y≈510+ without giving the post_corridor
        # milestone too early.
        if next_idx >= 4 and y >= 470.0:
            if hasattr(self, "reward_manager") and self.reward_manager is not None:
                self.reward_manager.add("toward_post_corridor", 0.02)
            return 0.02

        zone = route_zones[next_idx]
'''

if old not in text:
    raise SystemExit("Could not find route next_idx block.")

text = text.replace(old, new, 1)

# ------------------------------------------------------------
# 3. Add debug for route reset if reset function contains route state.
# This does not change behavior, but it helps verify route state only resets
# at episode boundaries.
# ------------------------------------------------------------
reset_pattern = r"(self\.route_progress_level\s*=\s*0\s*\n\s*self\.route_zones_reached\s*=\s*set\(\)\s*)"

if "[route_reset] route progress reset" not in text:
    text = re.sub(
        reset_pattern,
        r"\\1\n        print(\"[route_reset] route progress reset\")\n",
        text,
        count=1,
    )

p.write_text(text)
print("Patched Stage 01 stable route learning.")
