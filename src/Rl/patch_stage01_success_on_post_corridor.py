from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_stage01_success_on_post_corridor")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''            self.route_zones_reached.add(name)
            self.route_progress_level = next_idx + 1

            zone_reward = float(forced_route_reward(name))
'''

new = '''            self.route_zones_reached.add(name)
            self.route_progress_level = next_idx + 1

            if str(name) == "post_corridor":
                self.stage01_post_corridor_reached = True

            zone_reward = float(forced_route_reward(name))
'''

if old not in text and "self.stage01_post_corridor_reached = True" not in text:
    raise SystemExit("Could not find route_progress_reward zone update block.")

if "self.stage01_post_corridor_reached = True" not in text:
    text = text.replace(old, new, 1)

if "self.stage01_post_corridor_reached = False" not in text:
    marker = "self.route_progress_level = 0"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find route_progress_level reset marker.")
    line_end = text.find("\\n", idx)
    text = text[:line_end+1] + "        self.stage01_post_corridor_reached = False\\n" + text[line_end+1:]

marker = '''        if self.use_action_prior_advice:
'''

insert = '''        # Stage 01 success condition:
        # Once the route reaches post_corridor, end the episode as a success.
        task_name_success = str(getattr(self, "task_name", "") or "").lower()
        task_cfg_success = str(getattr(getattr(self, "task_config", None), "name", "") or "").lower()
        if (
            getattr(self, "stage01_post_corridor_reached", False)
            and ("stage_01" in task_name_success or "stage_01" in task_cfg_success or "navigation" in task_name_success)
        ):
            reward += 250.0
            done = True
            info["stage01_success"] = True
            info["success"] = True
            if hasattr(self, "reward_manager") and self.reward_manager is not None:
                self.reward_manager.add("stage01_post_corridor_success", 250.0)

'''

if "stage01_post_corridor_success" not in text:
    if marker not in text:
        raise SystemExit("Could not find action-prior hook marker for success insertion.")
    text = text.replace(marker, insert + marker, 1)

p.write_text(text)
print("Patched Stage 01 success when post_corridor is reached.")
