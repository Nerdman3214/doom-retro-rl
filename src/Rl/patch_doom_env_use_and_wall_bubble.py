from pathlib import Path
import re


ENV_PATH = Path("env/doom_env.py")
BACKUP_PATH = Path("env/doom_env.py.before_use_wall_patch")

text = ENV_PATH.read_text()

if not BACKUP_PATH.exists():
    BACKUP_PATH.write_text(text)
    print(f"Backup saved to {BACKUP_PATH}")


# ------------------------------------------------------------
# 1. Replace the broken if action == "use" block.
# ------------------------------------------------------------
# Bug:
#   meaningful_use was checked before it was assigned.
#
# Fix:
#   calculate door_visible, door_centered, meaningful_use first,
#   then use meaningful_use for secret memory and rewards.
# ------------------------------------------------------------

old_use_pattern = r'''        if action == "use":
            # Only count "use" as successful when the vision system sees a likely
            # door/button in front of the agent\. This prevents use-spam from being
            # treated as door mastery\.
            door_visible = vision\.get\("door_visible", False\)
            door_centered = vision\.get\("door_centered", False\)

            x = game_state\.get\("x"\)
            y = game_state\.get\("y"\)

            if x is not None and y is not None:
                use_tile = \(int\(float\(x\) // 64\), int\(float\(y\) // 64\)\)

                if meaningful_use and use_tile not in self\.secret_use_locations:
                    self\.secret_use_locations\.add\(use_tile\)
                    reward \+= 3\.0
                    self\.reward_manager\.add\("new_use_location_discovered", 3\.0\)

                    print\(f"\[memory\] useful use location discovered: \{use_tile\}"\)

            meaningful_use = \(
                self\.curriculum_stage in \[2, 5, 6, 7\]
                and door_visible
                and door_centered
                and \(
                    self\.stuck_counter >= 2
                    or self\.wall_contact_steps >= 1
                    or distance_moved < 1\.0
                \)
            \)

            if meaningful_use:
                self\.door_interaction_count \+= 1
                reward \+= 1\.0
                self\.reward_manager\.add\("meaningful_door_use", 1\.0\)
            else:
                reward \+= self\.add_penalty\("use_spam_penalty", -1\.0\)
'''

new_use_block = '''        if action == "use":
            # -------------------------------------------------
            # Door / button / secret use logic
            # -------------------------------------------------
            # IMPORTANT:
            # meaningful_use must be calculated BEFORE we check it.
            #
            # This fixes:
            # UnboundLocalError: cannot access local variable
            # 'meaningful_use' where it is not associated with a value.
            #
            # The agent only gets credit for pressing use when:
            #   - use is allowed in the current stage
            #   - a door/button is likely visible and centered
            #   - the agent is close enough / slowed enough for use to matter
            door_visible = vision.get("door_visible", False)
            door_centered = vision.get("door_centered", False)

            # The learned scene classifier can also help identify doors/buttons.
            if scene_label == "door_or_button" and scene_confidence >= 0.45:
                door_visible = True

                # If the classifier sees a door/button, treat it as centered enough
                # for now. Later you can improve this with object localization.
                door_centered = True

            meaningful_use = (
                self.curriculum_stage in [2, 3, 5, 6, 7]
                and door_visible
                and door_centered
                and (
                    self.stuck_counter >= 2
                    or self.wall_contact_steps >= 1
                    or distance_moved < 2.0
                    or scene_label == "door_or_button"
                )
            )

            if meaningful_use:
                self.door_interaction_count += 1
                reward += 1.0
                self.reward_manager.add("meaningful_door_use", 1.0)

                x = game_state.get("x")
                y = game_state.get("y")

                if x is not None and y is not None:
                    use_tile = (int(float(x) // 64), int(float(y) // 64))

                    # Remember useful use locations across episodes.
                    # This helps with secret doors/buttons because the agent
                    # can discover that use worked at this approximate tile.
                    if use_tile not in self.secret_use_locations:
                        self.secret_use_locations.add(use_tile)

                        reward += 3.0
                        self.reward_manager.add("new_use_location_discovered", 3.0)

                        print(f"[memory] useful use location discovered: {use_tile}")

            else:
                reward += self.add_penalty("use_spam_penalty", -1.0)
'''

text, count = re.subn(old_use_pattern, new_use_block, text, count=1, flags=re.S)

if count != 1:
    raise RuntimeError(
        "Could not replace the old action == 'use' block. "
        "Your file may have changed. Search for: if action == \"use\":"
    )

print("Fixed action == 'use' block.")


# ------------------------------------------------------------
# 2. Add pre-action wall bubble safety.
# ------------------------------------------------------------
# Your current wall bubble block is after perform_action().
# That means it changes action after Doom already received the keypress.
#
# This inserts a light pre-action version before melee tracking.
# ------------------------------------------------------------

insert_point = '''        # Track melee streak after final helper choice.
        if action == "melee_attack":'''

pre_action_bubble = '''        # -----------------------------------------------------
        # Pre-action wall bubble safety
        # -----------------------------------------------------
        # This runs BEFORE perform_action(), so it affects the real keypress.
        #
        # The post-action wall bubble is still useful for rewards/debugging,
        # but action overrides must happen before Doom receives input.
        pre_wall_bubble = self.wall_bubble_state(
            wall_info=pre_wall_info,
            distance_moved=0.0,
            motion=0.0,
            action=action,
        )

        before_wall_bubble = action
        action = self.wall_bubble_action(action, pre_wall_bubble)

        if action != before_wall_bubble:
            print(
                f"[override] pre_wall_bubble: {before_wall_bubble} -> {action} "
                f"level={pre_wall_bubble['level']} "
                f"dir={pre_wall_bubble['direction']} "
                f"L={pre_wall_bubble['left']:.2f} "
                f"F={pre_wall_bubble['front']:.2f} "
                f"R={pre_wall_bubble['right']:.2f}"
            )

        # Track melee streak after final helper choice.
        if action == "melee_attack":'''

if insert_point not in text:
    raise RuntimeError("Could not find melee tracking insert point.")

# Only insert if it is not already there.
if "pre_wall_bubble" not in text:
    text = text.replace(insert_point, pre_action_bubble, 1)
    print("Inserted pre-action wall bubble safety block.")
else:
    print("Pre-action wall bubble block already exists; skipped insertion.")


# ------------------------------------------------------------
# 3. Fix final safety fallback.
# ------------------------------------------------------------
# In Stage 3, falling back to move_forward can be bad near walls.
# Use move_backward if wall bubble says red/orange front danger.
# ------------------------------------------------------------

old_final_safety = '''        if action not in self.get_allowed_actions():
            fixed_action = "move_forward"
            print(
                f"[override] final_safety: {action} -> {fixed_action} "
                f"because stage={self.curriculum_stage} allowed={self.get_allowed_actions()}"
            )
            action = fixed_action
'''

new_final_safety = '''        if action not in self.get_allowed_actions():
            # Safer fallback than always moving forward.
            if pre_wall_info.get("front_wall", False) or pre_wall_info.get("front_ratio", 0.0) > 0.45:
                fixed_action = "move_backward"
            else:
                fixed_action = "move_forward"

            print(
                f"[override] final_safety: {action} -> {fixed_action} "
                f"because stage={self.curriculum_stage} allowed={self.get_allowed_actions()}"
            )
            action = fixed_action
'''

if old_final_safety in text:
    text = text.replace(old_final_safety, new_final_safety, 1)
    print("Updated final safety fallback.")
else:
    print("Final safety fallback block not found or already updated.")


# ------------------------------------------------------------
# 4. Make Stage 3 use action count as a movement action too.
# ------------------------------------------------------------

old_stage3_movement = '''            movement_actions = [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "shoot",
                "swap_weapon",
            ]'''

new_stage3_movement = '''            movement_actions = [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "shoot",
                "use",
                "swap_weapon",
            ]'''

if old_stage3_movement in text:
    text = text.replace(old_stage3_movement, new_stage3_movement, 1)
    print("Added use to Stage 3 movement/action list.")
else:
    print("Stage 3 movement action list not found or already updated.")


ENV_PATH.write_text(text)

print(f"Patched {ENV_PATH}")
print("Now run:")
print("  python -m py_compile env/doom_env.py")
print("  python training/train_rl_agent.py")