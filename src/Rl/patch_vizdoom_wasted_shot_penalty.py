from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_wasted_shot_penalty")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add flags.
flag_block = '''        # Strong shooting discipline test.
        self.wasted_shot_penalty = -60.0
        self.poor_aim_shot_penalty = -24.0
        self.good_shot_reward = 8.0
'''

if "self.wasted_shot_penalty = -60.0" not in text:
    marker_options = [
        "        self.curriculum_movement_plus_shooting = True\n",
        "        self.rule_enforcer_reward_scale = 1.0\n",
    ]
    inserted = False
    for marker in marker_options:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break
    if not inserted:
        print("Warning: could not insert wasted-shot flags.")

# Insert penalty after action_name exists and before reward return.
needle = '''        info["rule_enforcer"] = rule_result
'''

insert = '''        # -----------------------------------------------------
        # Strong wasted-shot penalty
        # -----------------------------------------------------
        try:
            enemy_visible = bool(post_game_state.get("enemy_visible", False))
            enemy_centered = bool(post_game_state.get("enemy_centered", False))
            ammo = int(post_game_state.get("ammo", 0) or 0)

            if action_name == "shoot":
                if enemy_visible and enemy_centered and ammo > 0:
                    reward += float(getattr(self, "good_shot_reward", 8.0))
                    info["shot_quality"] = "good_centered"
                elif enemy_visible and ammo > 0:
                    reward += float(getattr(self, "poor_aim_shot_penalty", -24.0))
                    info["shot_quality"] = "poor_aim"
                else:
                    reward += float(getattr(self, "wasted_shot_penalty", -60.0))
                    info["shot_quality"] = "wasted"
        except Exception:
            pass

'''

if insert not in text:
    if needle in text:
        text = text.replace(needle, needle + insert, 1)
    else:
        print("Warning: could not find rule_enforcer info marker. Wasted-shot insert skipped.")

p.write_text(text)
print("Patched ViZDoom wasted-shot penalties.")
