from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_force_done_on_post_corridor")
backup.write_text(text)
print(f"Backup saved to {backup}")

# 1) When post_corridor is reached, set a strong forced-done flag.
old = '''            if str(name) == "post_corridor":
                self.stage01_post_corridor_reached = True

            zone_reward = float(forced_route_reward(name))
'''

new = '''            if str(name) == "post_corridor":
                self.stage01_post_corridor_reached = True
                self.stage01_force_done = True

            zone_reward = float(forced_route_reward(name))
'''

if old in text:
    text = text.replace(old, new, 1)
elif "self.stage01_force_done = True" not in text:
    old2 = '''            zone_reward = float(forced_route_reward(name))
'''
    new2 = '''            if str(name) == "post_corridor":
                self.stage01_post_corridor_reached = True
                self.stage01_force_done = True

            zone_reward = float(forced_route_reward(name))
'''
    if old2 not in text:
        raise SystemExit("Could not find route reward insertion point.")
    text = text.replace(old2, new2, 1)

# 2) Reset the forced-done flag at episode reset.
if "self.stage01_force_done = False" not in text:
    marker = "self.stage01_post_corridor_reached = False"
    if marker in text:
        idx = text.find(marker)
        line_end = text.find("\n", idx)
        text = text[:line_end+1] + "        self.stage01_force_done = False\n" + text[line_end+1:]
    else:
        marker2 = "self.route_progress_level = 0"
        idx = text.find(marker2)
        if idx == -1:
            raise SystemExit("Could not find reset marker.")
        line_end = text.find("\n", idx)
        text = text[:line_end+1] + "        self.stage01_post_corridor_reached = False\n        self.stage01_force_done = False\n" + text[line_end+1:]

# 3) Add helper block before returns.
success_block = '''        if getattr(self, "stage01_force_done", False):
            reward += 250.0
            done = True
            info["stage01_success"] = True
            info["success"] = True
            if hasattr(self, "reward_manager") and self.reward_manager is not None:
                self.reward_manager.add("stage01_post_corridor_success", 250.0)
            print("[stage01_success] post_corridor reached -> done=True")

'''

if "[stage01_success] post_corridor reached -> done=True" not in text:
    return_patterns = [
        "        return obs, reward, done, info",
        "        return observation, reward, done, info",
        "        return state, reward, done, info",
        "        return obs, reward, terminated, truncated, info",
        "        return observation, reward, terminated, truncated, info",
    ]

    replaced = False
    for pat in return_patterns:
        if pat in text:
            if "terminated, truncated" in pat:
                block = '''        if getattr(self, "stage01_force_done", False):
            reward += 250.0
            terminated = True
            truncated = False
            info["stage01_success"] = True
            info["success"] = True
            if hasattr(self, "reward_manager") and self.reward_manager is not None:
                self.reward_manager.add("stage01_post_corridor_success", 250.0)
            print("[stage01_success] post_corridor reached -> terminated=True")

'''
                text = text.replace(pat, block + pat, 1)
            else:
                text = text.replace(pat, success_block + pat, 1)
            replaced = True
            break

    if not replaced:
        raise SystemExit("Could not find step return pattern. Paste the final 80 lines of step().")

p.write_text(text)
print("Forced Stage 01 done on post_corridor.")
