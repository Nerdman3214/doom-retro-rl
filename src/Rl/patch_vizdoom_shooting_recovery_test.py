from pathlib import Path
import re

# ---------------------------------------------------------
# 1. Patch rule enforcer: strong but not PPO-destroying
# ---------------------------------------------------------

p = Path("director/doom_rule_enforcer.py")
text = p.read_text()

backup = Path("director/doom_rule_enforcer.py.backup_before_shooting_recovery_test")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Change 6x negative to 3x for navigation, keep shooting discipline strong elsewhere.
# Current 6x is producing massive value_loss and very negative reward.
text = text.replace(
    "return -abs(float(amount)) * 6.0",
    "return -abs(float(amount)) * 3.0"
)

# Clip tighter so PPO can still learn.
text = text.replace(
    "return max(-120.0, min(120.0, float(reward)))",
    "return max(-60.0, min(120.0, float(reward)))"
)

# Make blind shots bad, but not so bad that agent never explores shooting.
text = text.replace("self._neg(10.0)", "self._neg(6.0)")
text = text.replace("self._neg(8.0)", "self._neg(5.0)")

# Make centered enemy shooting clearly worth trying.
text = text.replace(
    "reward += 8.0\n                events.append(\"good_shoot_centered_enemy\")",
    "reward += 18.0\n                events.append(\"good_shoot_centered_enemy\")"
)

text = text.replace(
    "reward += 3.0\n                events.append(\"okay_shoot_visible_enemy\")",
    "reward += 6.0\n                events.append(\"okay_shoot_visible_enemy\")"
)

p.write_text(text)
print("Patched rule enforcer: negatives still strong, but PPO less likely to collapse.")


# ---------------------------------------------------------
# 2. Patch vizdoom env: shooting exploration when enemy visible
# ---------------------------------------------------------

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_shooting_exploration")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add flags.
flag_block = '''        # Shooting exploration test:
        # Keep PPO in control, but encourage occasional shoot sampling when enemy is visible.
        self.force_enemy_shoot_exploration = True
        self.enemy_shoot_explore_every = 12
        self.enemy_centered_shoot_explore_every = 5
        self.wasted_shot_penalty = -24.0
        self.poor_aim_shot_penalty = -12.0
        self.good_shot_reward = 18.0
'''

if "self.force_enemy_shoot_exploration = True" not in text:
    marker_options = [
        "        self.curriculum_movement_plus_shooting = True\n",
        "        self.exit_first_mode = True\n",
        "        self.rule_enforcer = DoomRuleEnforcer()\n",
    ]
    inserted = False
    for marker in marker_options:
        if marker in text:
            text = text.replace(marker, marker + flag_block, 1)
            inserted = True
            break
    if not inserted:
        print("Warning: could not insert shooting exploration flags.")

# Make sure existing harsh direct penalties are not still -60/-24.
text = text.replace("self.wasted_shot_penalty = -60.0", "self.wasted_shot_penalty = -24.0")
text = text.replace("self.poor_aim_shot_penalty = -24.0", "self.poor_aim_shot_penalty = -12.0")
text = text.replace("self.good_shot_reward = 8.0", "self.good_shot_reward = 18.0")

# Insert exploration after action_name is likely known and post_game_state exists.
needle = '''        rule_result = self.rule_enforcer.evaluate(
            game_state=post_game_state,
            action_name=action_name,
            mission_update=mission_update,
            level_guide=getattr(self, "level_guide", None),
        )
'''

insert = '''        # -----------------------------------------------------
        # Enemy-visible shooting exploration
        # -----------------------------------------------------
        # This is a temporary test. PPO still usually controls action,
        # but if the agent never shoots, it never learns shooting feedback.
        try:
            enemy_visible_for_explore = bool(post_game_state.get("enemy_visible", False))
            enemy_centered_for_explore = bool(post_game_state.get("enemy_centered", False))
            ammo_for_explore = int(post_game_state.get("ammo", 0) or 0)
            step_for_explore = int(getattr(self, "step_count", getattr(self, "episode_step", 0)) or 0)

            if (
                getattr(self, "force_enemy_shoot_exploration", False)
                and ammo_for_explore > 0
                and enemy_visible_for_explore
                and action_name != "shoot"
            ):
                if enemy_centered_for_explore and step_for_explore % int(getattr(self, "enemy_centered_shoot_explore_every", 5)) == 0:
                    action_name = "shoot"
                    info["shoot_explore"] = "enemy_centered"
                elif step_for_explore % int(getattr(self, "enemy_shoot_explore_every", 12)) == 0:
                    action_name = "shoot"
                    info["shoot_explore"] = "enemy_visible"
        except Exception:
            pass

'''

if insert not in text:
    if needle in text:
        text = text.replace(needle, insert + "\n" + needle, 1)
    else:
        print("Warning: rule_result marker not found; shooting exploration not inserted.")

p.write_text(text)
print("Patched ViZDoom env: shooting exploration enabled.")
