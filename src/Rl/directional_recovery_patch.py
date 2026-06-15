from pathlib import Path

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_directional_recovery_patch")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add a combined sensory + vision directional recovery helper
# ---------------------------------------------------------

helper = r'''
    def directional_recovery_action(
        self,
        action,
        wall_info=None,
        sensory_state=None,
        scene_label=None,
        scene_confidence=0.0,
        distance_moved=None,
        motion=None,
    ):
        """
        Combined sensory + vision recovery.

        Instead of blindly doing:
            move_backward -> turn_left/right

        this uses obstacle direction:
            obstacle left  -> turn/strafe right
            obstacle right -> turn/strafe left
            obstacle front -> rotate toward more open side
            boxed in       -> back up briefly, then rotate toward open side
        """

        wall_info = wall_info or {}
        sensory_state = sensory_state or {}

        left = float(wall_info.get("left_ratio", 0.0) or 0.0)
        front = float(wall_info.get("front_ratio", 0.0) or 0.0)
        right = float(wall_info.get("right_ratio", 0.0) or 0.0)

        situation = sensory_state.get("situation") or "normal_navigation"
        recommended = sensory_state.get("recommended_action")

        obstacle_label = scene_label in [
            "front_wall",
            "obstacle",
            "boundary_or_stuck_wall",
            "half_wall_rail",
            "side_rail",
        ]

        has_motion_data = distance_moved is not None and motion is not None

        low_translation = (
            has_motion_data
            and action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]
            and float(distance_moved or 0.0) <= 1.0
            and float(motion or 0.0) < 2.0
        )

        recovery_needed = (
            situation in [
                "stuck_or_looping",
                "front_blocked",
                "spawn_wall_zone",
                "secret_side_area",
                "right_route_area",
                "half_wall_rail",
                "side_rail",
            ]
            or obstacle_label
            or front >= 0.45
            or left >= 0.45
            or right >= 0.45
            or self.stuck_counter >= 2
            or self.wall_contact_steps >= 1
            or low_translation
            or recommended in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        )

        if not recovery_needed:
            return action

        # Which side is more blocked?
        left_blocked = left >= 0.40
        right_blocked = right >= 0.40
        front_blocked = front >= 0.42 or situation == "front_blocked"

        # Boxed-in case: back up briefly, then turn toward clearer side.
        boxed_in = front >= 0.60 and left >= 0.45 and right >= 0.45

        if boxed_in:
            cycle = self._step_count % 6

            if cycle in [0, 1]:
                return "move_backward"

            return "turn_left" if right < left else "turn_right"

        # Obstacle strongly on the left: turn/strafe right.
        if left_blocked and left > right + 0.08:
            if action == "strafe_left":
                return "strafe_right"

            # Turn away first, then strafe away sometimes.
            if self._step_count % 3 == 0:
                return "strafe_right"

            return "turn_right"

        # Obstacle strongly on the right: turn/strafe left.
        if right_blocked and right > left + 0.08:
            if action == "strafe_right":
                return "strafe_left"

            if self._step_count % 3 == 0:
                return "strafe_left"

            return "turn_left"

        # Front obstacle / half-wall / rail: rotate toward the more open side.
        if front_blocked:
            if left < right:
                return "turn_left"
            if right < left:
                return "turn_right"

            # Tie-breaker so it does not oscillate forever.
            return "turn_left" if (self._step_count // 4) % 2 == 0 else "turn_right"

        # Side rail but no front block: avoid scraping into the blocked side.
        if scene_label == "side_rail":
            if left > right:
                return "strafe_right"
            if right > left:
                return "strafe_left"

        # Last-resort loop escape: prefer turning away from strongest side.
        if situation == "stuck_or_looping" or self.stuck_counter >= 2:
            if left > right:
                return "turn_right"
            if right > left:
                return "turn_left"

            return "move_backward" if self._step_count % 4 == 0 else "turn_right"

        return action

'''

if "def directional_recovery_action(" not in text:
    marker = "    def reset("
    if marker not in text:
        raise SystemExit("Could not find reset() marker.")
    text = text.replace(marker, helper + "\n" + marker, 1)

# ---------------------------------------------------------
# 2. Patch wall_bubble_action to turn opposite obstacle direction
# ---------------------------------------------------------

old_wall_bubble = r'''    def wall_bubble_action(self, action, bubble):
        """
        Safer wall-bubble correction.

        Orange does not mean panic. It only nudges the agent away.
        Red means blocked/bumping and should force a stronger escape.
        """
        level = bubble["level"]
        direction = bubble["direction"]

        if level == "green":
            return action

        # Red = blocked or actually bumping.
        if level == "red":
            if direction == "front":
                if action == "move_forward":
                    return "move_backward"

                # Rotate away after backing up.
                if bubble["left"] > bubble["right"]:
                    return "turn_right"
                return "turn_left"

            if direction == "left":
                if action in ["move_forward", "strafe_left"]:
                    return "strafe_right"
                return "turn_right"

            if direction == "right":
                if action in ["move_forward", "strafe_right"]:
                    return "strafe_left"
                return "turn_left"

        # Orange = close to wall, but not necessarily stuck.
        # Do not override turning/use/shoot unless it is clearly bad.
        if bubble["level"] == "orange":
            if action == "move_forward" and bubble["direction"] == "front" and bubble["front"] >= 0.60:
                return "move_backward"

            if action == "strafe_left" and bubble["direction"] == "left" and bubble["left"] >= 0.75:
                return "strafe_right"

            if action == "strafe_right" and bubble["direction"] == "right" and bubble["right"] >= 0.75:
                return "strafe_left"

            return action
'''

new_wall_bubble = r'''    def wall_bubble_action(self, action, bubble):
        """
        Directional wall-bubble correction.

        Key change:
        - Do not always back up from front pressure.
        - Turn toward the more open side.
        - Turn/strafe opposite side obstacles.
        """

        level = bubble["level"]
        direction = bubble["direction"]

        left = float(bubble.get("left", 0.0) or 0.0)
        front = float(bubble.get("front", 0.0) or 0.0)
        right = float(bubble.get("right", 0.0) or 0.0)

        if level == "green":
            return action

        # Pick the more open side.
        open_turn = "turn_left" if left < right else "turn_right"
        open_strafe = "strafe_left" if left < right else "strafe_right"

        if level == "red":
            if direction == "front":
                # Only back up sometimes. Mostly rotate toward open space.
                if action == "move_forward":
                    return open_turn

                if action == "move_backward" and front >= 0.60:
                    return open_turn

                return open_turn

            if direction == "left":
                if action in ["move_forward", "strafe_left", "turn_left"]:
                    return "turn_right"
                return "strafe_right"

            if direction == "right":
                if action in ["move_forward", "strafe_right", "turn_right"]:
                    return "turn_left"
                return "strafe_left"

        if level == "orange":
            if direction == "front" and front >= 0.58:
                if action == "move_forward":
                    return open_turn

            if direction == "left" and left >= 0.55:
                if action in ["strafe_left", "turn_left", "move_forward"]:
                    return "turn_right"

            if direction == "right" and right >= 0.55:
                if action in ["strafe_right", "turn_right", "move_forward"]:
                    return "turn_left"

            return action

        return action
'''

if old_wall_bubble in text:
    text = text.replace(old_wall_bubble, new_wall_bubble)
else:
    print("Warning: exact wall_bubble_action block not found. Skipping full replacement.")

# ---------------------------------------------------------
# 3. Insert pre-action directional recovery after pre_wall_bubble
# ---------------------------------------------------------

needle = r'''        if action != before_wall_bubble:
            print(
                f"[override] pre_wall_bubble: {before_wall_bubble} -> {action} "
                f"level={pre_wall_bubble['level']} "
                f"dir={pre_wall_bubble['direction']} "
                f"L={pre_wall_bubble['left']:.2f} "
                f"F={pre_wall_bubble['front']:.2f} "
                f"R={pre_wall_bubble['right']:.2f}"
            )
'''

insert = r'''
        # -----------------------------------------------------
        # Directional sensory + vision recovery
        # -----------------------------------------------------
        # This combines wall ratios, sensory situation, and scene label.
        # It turns opposite the obstacle direction instead of blindly backing up.
        before_directional_recovery = action

        pre_sensory_state_for_recovery = {
            "situation": pre_game_state.get("sensory_situation", "normal_navigation"),
            "recommended_action": pre_game_state.get("sensory_recommended_action"),
        }

        action = self.directional_recovery_action(
            action=action,
            wall_info=pre_wall_info,
            sensory_state=pre_sensory_state_for_recovery,
            scene_label=pre_scene_label,
            scene_confidence=pre_scene_confidence,
            distance_moved=None,
            motion=None,
        )

        if action != before_directional_recovery:
            print(
                f"[override] directional_recovery: "
                f"{before_directional_recovery} -> {action} "
                f"scene={pre_scene_label} "
                f"L={pre_wall_info.get('left_ratio', 0.0):.2f} "
                f"F={pre_wall_info.get('front_ratio', 0.0):.2f} "
                f"R={pre_wall_info.get('right_ratio', 0.0):.2f}"
            )

'''

if insert not in text:
    text = text.replace(needle, needle + insert, 1)

# ---------------------------------------------------------
# 4. Make hard_stuck_escape directional too
# ---------------------------------------------------------

old_hard_stuck = r'''        if self.stuck_counter >= 8 and not self.sensory_emergency_active:
            cycle = self._step_count % 8

            before = action

            if cycle in [0, 1]:
                action = "move_backward"
            elif cycle in [2, 3]:
                action = "turn_right"
            elif cycle == 4:
                action = "strafe_right"
            elif cycle == 5:
                action = "turn_left"
            else:
                action = "move_forward"

            print(f"[override] hard_stuck_escape: {before} -> {action}")
'''

new_hard_stuck = r'''        if self.stuck_counter >= 8 and not self.sensory_emergency_active:
            before = action

            action = self.directional_recovery_action(
                action=action,
                wall_info=pre_wall_info,
                sensory_state={"situation": "stuck_or_looping"},
                scene_label=pre_scene_label,
                scene_confidence=pre_scene_confidence,
                distance_moved=None,
                motion=None,
            )

            print(f"[override] hard_stuck_directional_escape: {before} -> {action}")
'''

if old_hard_stuck in text:
    text = text.replace(old_hard_stuck, new_hard_stuck)
else:
    print("Warning: hard_stuck_escape block not found. Skipping.")

p.write_text(text)
print("Applied directional sensory+vision recovery patch.")
