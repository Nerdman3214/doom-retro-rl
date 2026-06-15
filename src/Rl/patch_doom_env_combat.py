from pathlib import Path
import re


ENV_PATH = Path("env/doom_env.py")
BACKUP_PATH = Path("env/doom_env.py.before_combat_patch")

text = ENV_PATH.read_text()

if not BACKUP_PATH.exists():
    BACKUP_PATH.write_text(text)
    print(f"Backup saved to {BACKUP_PATH}")


def replace_method(source: str, method_name: str, replacement: str) -> str:
    pattern = rf"\n    def {method_name}\(.*?\n(?=\n    def |\n    # ---------------------------------------------------------|\n    @|\nclass |\Z)"
    updated, count = re.subn(pattern, "\n" + replacement.rstrip() + "\n", source, count=1, flags=re.S)

    if count != 1:
        raise RuntimeError(f"Could not replace method: {method_name}")

    print(f"Replaced method: {method_name}")
    return updated


AIM_ASSIST_ACTION = r'''
    def aim_assist_action(self, action, frame, enemy_visible, enemy_centered, ammo):
        """
        Aim helper only.

        This function is intentionally NOT allowed to choose shoot anymore.
        The old version returned shoot from too many branches, which caused:
        - random shooting
        - ammo drain
        - valid_shot_count going up without kills
        - shoot/melee/move_forward override loops

        Shooting is handled only by vision_combat_action().
        """
        if self.curriculum_stage < 3:
            return action

        if self.dead_target_ignore_steps > 0:
            self.dead_target_ignore_steps -= 1
            return action

        if not enemy_visible:
            return action

        ammo = int(ammo or 0)

        # No ammo means do not aim-shoot. Back up / strafe instead.
        if ammo <= 0:
            if self._step_count % 2 == 0:
                return "move_backward"
            return "strafe_right"

        aim_error, confidence = self._enemy_horizontal_error(frame)

        # If the old color detector is weak/noisy, do not shoot.
        # The learned scene classifier decides enemy existence.
        if confidence < 12:
            return "move_backward"

        if aim_error < -0.18:
            self.last_aim_assist_step = self._step_count
            return "turn_left"

        if aim_error > 0.18:
            self.last_aim_assist_step = self._step_count
            return "turn_right"

        # Already roughly aimed; leave shoot decision to vision_combat_action().
        return action
'''


VISION_COMBAT_ACTION = r'''
    def vision_combat_action(self, action, scene_label, scene_confidence, health, ammo):
        """
        The ONLY helper that may turn an action into shoot.

        Rules:
        - Never shoot before Stage 3.
        - Never shoot with no ammo.
        - Never shoot while stuck/corner trapped.
        - Only shoot when the learned scene classifier is confident.
        - Fire short bursts, then move to keep distance.
        """
        if self.curriculum_stage < 3:
            return action

        # Corner/stuck escape beats combat.
        if self.corner_trap_steps >= 15 or self.stuck_counter >= 5:
            return action

        enemy_seen = (
            scene_label == "enemy"
            and scene_confidence >= 0.80
        )

        if not enemy_seen:
            return action

        ammo = int(ammo or 0)
        health = int(health or 0)

        # Never choose shoot with no ammo.
        if ammo <= 0:
            if health <= 50:
                return "move_backward"
            return "strafe_right"

        # Low health means survival first.
        if health <= 40:
            if self._step_count % 3 == 0:
                return "move_backward"
            return "strafe_right"

        # Controlled burst pattern:
        # 2 shoot steps, then movement/spacing.
        cycle = self._step_count % 8

        if cycle in [0, 1]:
            return "shoot"

        if cycle in [2, 3]:
            return "strafe_left"

        if cycle in [4, 5]:
            return "move_backward"

        return action
'''


SANITIZE_ACTION = r'''
    def sanitize_action(self, action, game_state, enemy_visible):
        """
        Final action gate for the current curriculum stage.

        This blocks illegal/impossible actions, but it does NOT create new
        combat behavior. In particular, it must not convert swap_weapon or
        no-ammo cases back into shoot.
        """
        config = self.get_stage_config()

        ammo = int(game_state.get("ammo", 0) or 0)
        current_weapon = str(game_state.get("weapon", "")).lower()

        is_melee_weapon = (
            "fist" in current_weapon
            or "chainsaw" in current_weapon
            or "ripter" in current_weapon
            or "melee" in current_weapon
        )

        if action == "use" and not config.get("allow_use", False):
            return "move_forward"

        if action == "shoot":
            if not config.get("allow_shoot", False):
                return "move_forward"

            # No ammo means no shoot. Do not convert back into shoot later.
            if ammo <= 0:
                if is_melee_weapon and enemy_visible and config.get("allow_melee", False):
                    return "melee_attack"
                return "move_backward"

            if config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
                return "move_forward"

            return action

        if action == "melee_attack":
            if not config.get("allow_melee", False):
                return "move_forward"

            if not enemy_visible:
                return "move_forward"

            return action

        if action == "swap_weapon":
            # Swap is allowed, but not repeatedly.
            # Never turn swap_weapon into shoot here.
            if self.consecutive_swap_steps >= 2:
                return "move_backward" if enemy_visible else "move_forward"
            return action

        return action
'''


text = replace_method(text, "aim_assist_action", AIM_ASSIST_ACTION)
text = replace_method(text, "vision_combat_action", VISION_COMBAT_ACTION)
text = replace_method(text, "sanitize_action", SANITIZE_ACTION)


# ------------------------------------------------------------
# Add pre-action scene classifier block.
# ------------------------------------------------------------

old_pre = '''        pre_frame = self.observer.get_frame()
        pre_game_state = self.observer.get_game_state()
        pre_vision = self.detect_vision(pre_frame)

        pre_enemy_visible = pre_vision["enemy_visible"]
        pre_enemy_centered = pre_vision["enemy_centered"]'''

new_pre = '''        pre_frame = self.observer.get_frame()
        pre_game_state = self.observer.get_game_state()
        pre_vision = self.detect_vision(pre_frame)

        # -----------------------------------------------------
        # Pre-action learned scene prediction
        # -----------------------------------------------------
        # This must happen BEFORE perform_action().
        # Any action override after perform_action() only changes the log,
        # not the keypress sent to Doom.
        pre_scene_label = "unclear"
        pre_scene_confidence = 0.0
        pre_scene_probs = {}

        if self.scene_predictor is not None:
            pre_scene_result = self.scene_predictor.predict(pre_frame)
            pre_scene_label = pre_scene_result["label"]
            pre_scene_confidence = pre_scene_result["confidence"]
            pre_scene_probs = pre_scene_result["probs"]

            if self._step_count % 25 == 0:
                print(
                    f"[vision_model] label={pre_scene_label} "
                    f"conf={pre_scene_confidence:.2f}"
                )

        pre_enemy_visible = pre_vision["enemy_visible"]
        pre_enemy_centered = pre_vision["enemy_centered"]

        # In Stage 3+, the learned classifier is the authority for enemy
        # existence. The older color detector is only used for left/right/center
        # aiming hints after the classifier confirms an enemy.
        if self.curriculum_stage >= 3 and self.scene_predictor is not None:
            classifier_enemy = (
                pre_scene_label == "enemy"
                and pre_scene_confidence >= 0.75
            )

            if classifier_enemy:
                pre_enemy_visible = True
                pre_enemy_centered = pre_vision["enemy_centered"]
            else:
                pre_enemy_visible = False
                pre_enemy_centered = False
                pre_vision["enemy_left"] = False
                pre_vision["enemy_right"] = False
                pre_vision["enemy_confidence"] = 0.0'''

if old_pre not in text:
    raise RuntimeError("Could not find pre-action perception block.")

text = text.replace(old_pre, new_pre, 1)
print("Inserted pre-action scene classifier block.")


# ------------------------------------------------------------
# Add scene fields into pre_game_state.
# ------------------------------------------------------------

old_state = '''        pre_game_state["wall_contact_steps"] = self.wall_contact_steps
        pre_game_state["action"] = action'''

new_state = '''        pre_game_state["wall_contact_steps"] = self.wall_contact_steps
        pre_game_state["scene_label"] = pre_scene_label
        pre_game_state["scene_confidence"] = pre_scene_confidence
        pre_game_state["scene_probs"] = pre_scene_probs
        pre_game_state["action"] = action'''

if old_state not in text:
    raise RuntimeError("Could not find pre_game_state action block.")

text = text.replace(old_state, new_state, 1)
print("Added scene fields to pre_game_state.")


# ------------------------------------------------------------
# Insert pre-action vision combat before melee/swap tracking.
# ------------------------------------------------------------

old_before_track = '''        # Track melee streak after final helper choice.
        if action == "melee_attack":'''

new_before_track = '''        # -----------------------------------------------------
        # Pre-action learned-vision safety and combat
        # -----------------------------------------------------
        # These happen BEFORE perform_action(), so they affect the real keypress.

        pre_wall_info = self._wall_direction_info(pre_frame)

        before_goal = action
        action = self.goal_assist_action(
            action=action,
            game_state=pre_game_state,
            wall_info=pre_wall_info,
            enemy_visible=pre_enemy_visible,
        )
        if action != before_goal:
            print(f"[override] goal_assist: {before_goal} -> {action}")

        if (
            pre_scene_label == "front_wall"
            and pre_scene_confidence >= 0.65
            and action == "move_forward"
        ):
            before = action
            action = "move_backward"
            print(f"[override] vision_front_wall: {before} -> {action}")

        # Corner/stuck escape beats combat. Do not shoot while trapped.
        if self.corner_trap_steps >= 20:
            before = action
            cycle = self.corner_trap_steps % 12

            if cycle in [0, 1, 2]:
                action = "move_backward"
            elif cycle in [3, 4, 5]:
                action = "turn_right"
            elif cycle in [6, 7]:
                action = "strafe_right"
            elif cycle in [8, 9]:
                action = "turn_left"
            else:
                action = "move_forward"

            print(
                f"[override] corner_escape: {before} -> {action} "
                f"corner_steps={self.corner_trap_steps}"
            )
        else:
            before_vision_combat = action
            action = self.vision_combat_action(
                action=action,
                scene_label=pre_scene_label,
                scene_confidence=pre_scene_confidence,
                health=pre_game_state.get("health", 100),
                ammo=pre_game_state.get("ammo", 0),
            )

            if action != before_vision_combat:
                print(
                    f"[override] vision_combat: "
                    f"{before_vision_combat} -> {action} "
                    f"label={pre_scene_label} conf={pre_scene_confidence:.2f}"
                )

        # Track melee streak after final helper choice.
        if action == "melee_attack":'''

if old_before_track not in text:
    raise RuntimeError("Could not find melee tracking insertion point.")

text = text.replace(old_before_track, new_before_track, 1)
print("Inserted pre-action vision combat block.")


# ------------------------------------------------------------
# Remove post-action vision safety block.
# This block is wrong because it changes action after perform_action().
# ------------------------------------------------------------

post_vision_pattern = r'''
        # -----------------------------------------------------
        # Vision-model safety hints
        # -----------------------------------------------------
        # The classifier is not controlling the whole agent.
        # It only prevents obviously bad actions.
        if scene_confidence >= 0\.65:
            .*?
        # -----------------------------------------------------
        # Corridor milestone
'''

replacement_post_vision = '''
        # -----------------------------------------------------
        # Corridor milestone
'''

text, count = re.subn(post_vision_pattern, replacement_post_vision, text, count=1, flags=re.S)

if count == 1:
    print("Removed post-action vision safety block.")
else:
    print("Post-action vision safety block not found or already removed.")


# ------------------------------------------------------------
# Remove later post-action vision_combat_action block if present.
# ------------------------------------------------------------

post_combat_pattern = r'''
        before_vision_combat = action
        action = self\.vision_combat_action\(
            action=action,
            scene_label=scene_label,
            scene_confidence=scene_confidence,
            health=health,
            ammo=ammo,
        \)

        if action != before_vision_combat:
            print\(
                f"\[override\] vision_combat: "
                f"\{before_vision_combat\} -> \{action\} "
                f"label=\{scene_label\} conf=\{scene_confidence:\.2f\}"
            \)

'''

text, count = re.subn(post_combat_pattern, "", text, count=1, flags=re.S)

if count == 1:
    print("Removed post-action vision_combat_action block.")
else:
    print("Post-action vision_combat_action block not found or already removed.")


# ------------------------------------------------------------
# Replace post-action scene predictor block with pre-action values.
# ------------------------------------------------------------

post_scene_pattern = r'''
        scene_label = "unclear"
        scene_confidence = 0\.0
        scene_probs = \{\}

        if self\.scene_predictor is not None:
            scene_result = self\.scene_predictor\.predict\(raw_frame\)
            scene_label = scene_result\["label"\]
            scene_confidence = scene_result\["confidence"\]
            scene_probs = scene_result\["probs"\]

            game_state\["scene_label"\] = scene_label
            game_state\["scene_confidence"\] = scene_confidence
            game_state\["scene_probs"\] = scene_probs

            if self\._step_count % 25 == 0:
                print\(f"\[vision_model\] label=\{scene_label\} conf=\{scene_confidence:\.2f\}"\)
'''

new_post_scene = '''
        # -----------------------------------------------------
        # Post-action scene state
        # -----------------------------------------------------
        # Use the pre-action prediction for reward/state consistency.
        # Do not modify action here; Doom already received the keypress.
        scene_label = pre_scene_label
        scene_confidence = pre_scene_confidence
        scene_probs = pre_scene_probs

        game_state["scene_label"] = scene_label
        game_state["scene_confidence"] = scene_confidence
        game_state["scene_probs"] = scene_probs
'''

text, count = re.subn(post_scene_pattern, new_post_scene, text, count=1, flags=re.S)

if count == 1:
    print("Replaced post-action scene predictor with pre-action values.")
else:
    print("Post-action scene predictor block not found or already replaced.")


# ------------------------------------------------------------
# Remove duplicated later goal assist block after wall_info.
# Goal assist now happens before perform_action().
# ------------------------------------------------------------

late_goal_pattern = r'''
                # -----------------------------------------------------
        # Goal steering
        # -----------------------------------------------------
        # This gives the agent a simple navigation bias:
        # rotate toward the goal, then move forward.
        # It does not run if walls are too close because wall escape
        # should take priority in tight spaces.
        before_goal = action
        action = self\.goal_assist_action\(
            action=action,
            game_state=game_state,
            wall_info=wall_info,
            enemy_visible=enemy_visible,
        \)

        if action != before_goal:
            print\(f"\[override\] goal_assist: \{before_goal\} -> \{action\}"\)

'''

text, count = re.subn(late_goal_pattern, "", text, count=1, flags=re.S)

if count == 1:
    print("Removed late post-action goal assist block.")
else:
    print("Late goal assist block not found or already removed.")


# ------------------------------------------------------------
# Make Stage 3 progression stricter.
# ------------------------------------------------------------

old_stage3 = '''        elif self.curriculum_stage == 3:
            behavior_ready = (
                self.valid_shot_count >= 3
                or self.enemy_kill_count >= 1
                or self.combat_survival_steps >= 100
            )'''

new_stage3 = '''        elif self.curriculum_stage == 3:
            # Do not advance just because the agent spammed shots.
            # Require either an actual kill, or survival with limited,
            # controlled shooting and some ammo remaining.
            behavior_ready = (
                self.enemy_kill_count >= 1
                or (
                    self.combat_survival_steps >= 250
                    and self.valid_shot_count >= 5
                    and self.valid_shot_count <= 80
                )
            )'''

if old_stage3 in text:
    text = text.replace(old_stage3, new_stage3, 1)
    print("Updated Stage 3 curriculum progression.")
else:
    print("Stage 3 progression block not found or already updated.")


# ------------------------------------------------------------
# Reduce Stage 3 shoot reward so it does not learn spam.
# ------------------------------------------------------------

text = text.replace(
    '''            if action == "shoot" and enemy_visible:
                reward += 1.00

                if enemy_centered:
                    reward += 2.00
                    self.enemy_engagement_count += 1
                    self.valid_shot_count += 1''',
    '''            if action == "shoot" and enemy_visible and ammo > 0:
                # Smaller reward to avoid ammo-spam learning.
                reward += 0.30

                if enemy_centered:
                    reward += 0.50
                    self.enemy_engagement_count += 1
                    self.valid_shot_count += 1''',
    1,
)

text = text.replace(
    '''        if enemy_visible and action == "shoot" and ammo > 0:
            reward += 0.75
            self.reward_manager.add("shoot_enemy_on_sight", 0.75)''',
    '''        if enemy_visible and action == "shoot" and ammo > 0:
            reward += 0.25
            self.reward_manager.add("shoot_enemy_on_sight", 0.25)''',
    1,
)


ENV_PATH.write_text(text)
print(f"Patched {ENV_PATH}")
print("Now run: python -m py_compile env/doom_env.py")