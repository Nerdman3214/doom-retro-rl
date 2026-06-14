from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_use_door_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

flag_block = '''        # Door/use discipline.
        self.good_use_reward = 10.0
        self.bad_use_spam_penalty = -2.0
'''

if "self.good_use_reward = 10.0" not in text:
    marker = "        self.good_shot_reward = 12.0\n"
    if marker in text:
        text = text.replace(marker, marker + flag_block, 1)
    else:
        marker = "        self.rule_enforcer_reward_scale = 1.0\n"
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
        else:
            print("Warning: could not insert use reward flags.")

needle = '''        info["rule_enforcer"] = rule_result
'''

insert = '''        # -----------------------------------------------------
        # Early use/door reward
        # -----------------------------------------------------
        try:
            door_visible = bool(post_game_state.get("door_visible", False))
            door_centered = bool(post_game_state.get("door_centered", False))
            near_use_point = bool(post_game_state.get("near_use_point", False))
            opened_door = bool(post_game_state.get("opened_door", False))
            scene_label = post_game_state.get("scene_label")

            door_like = (
                door_visible
                or door_centered
                or near_use_point
                or scene_label in ["door_or_button", "switch", "locked_door"]
            )

            if action_name == "use":
                if door_like:
                    reward += float(getattr(self, "good_use_reward", 10.0))
                    info["use_quality"] = "good_near_door"
                else:
                    reward += float(getattr(self, "bad_use_spam_penalty", -2.0))
                    info["use_quality"] = "spam"

            if opened_door:
                reward += 25.0
                info["opened_door"] = True
        except Exception:
            pass

'''

if insert not in text:
    if needle in text:
        text = text.replace(needle, needle + insert, 1)
    else:
        print("Warning: could not find rule_enforcer info marker. Use reward insert skipped.")

p.write_text(text)
print("Patched ViZDoom with early use/door reward.")
