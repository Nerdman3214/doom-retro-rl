from pathlib import Path
import re

p = Path("env/doom_env.py")
text = p.read_text()

# ---------------------------------------------------------
# 1. Add helper method inside DoomEnv
# ---------------------------------------------------------
helper = r'''
    def infer_half_wall_rail_state(
        self,
        scene_label,
        scene_confidence,
        wall_info=None,
        distance_moved=None,
        motion=None,
        action=None,
    ):
        """
        Detect corridor rails / half-walls.

        A half-wall/rail is not a normal full wall:
        - the screen may look partly open above it
        - but movement is blocked or heavily slowed
        - front wall ratio is often orange/red
        """

        wall_info = wall_info or {}

        front_ratio = float(wall_info.get("front_ratio", 0.0) or 0.0)
        left_ratio = float(wall_info.get("left_ratio", 0.0) or 0.0)
        right_ratio = float(wall_info.get("right_ratio", 0.0) or 0.0)

        side_ratio = max(left_ratio, right_ratio)

        has_motion_data = distance_moved is not None and motion is not None

        movement_action = action in [
            "move_forward",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]

        low_translation = (
            has_motion_data
            and movement_action
            and float(distance_moved or 0.0) <= 1.0
            and float(motion or 0.0) < 2.0
        )

        # Rail/half-wall signature:
        # front pressure is high enough to block movement, but not necessarily
        # a full-screen wall. This often happens on Doom corridor rails.
        front_rail_pressure = 0.52 <= front_ratio <= 0.78

        side_rail_pressure = 0.45 <= side_ratio <= 0.75

        fake_open = scene_label in [
            "open_path",
            "unclear",
            "normal_navigation",
        ]

        likely_front_half_wall = (
            fake_open
            and front_rail_pressure
            and (
                low_translation
                or self.stuck_counter >= 2
                or self.wall_contact_steps >= 1
                or getattr(self, "repeated_position_steps", 0) >= 2
            )
        )

        likely_side_rail = (
            fake_open
            and side_rail_pressure
            and not likely_front_half_wall
        )

        if likely_front_half_wall:
            return {
                "label": "half_wall_rail",
                "confidence": max(float(scene_confidence or 0.0), 0.88),
                "position": "front",
                "front_ratio": front_ratio,
                "side_ratio": side_ratio,
            }

        if likely_side_rail:
            return {
                "label": "side_rail",
                "confidence": max(float(scene_confidence or 0.0), 0.72),
                "position": "side",
                "front_ratio": front_ratio,
                "side_ratio": side_ratio,
            }

        return {
            "label": scene_label,
            "confidence": float(scene_confidence or 0.0),
            "position": None,
            "front_ratio": front_ratio,
            "side_ratio": side_ratio,
        }


    def half_wall_rail_reward(self, game_state, action):
        """
        Reward shaping for rails / half-walls.

        Front rail:
        - moving forward into it is very bad
        - backing up / turning / strafing away is good

        Side rail:
        - normal movement beside it is okay
        - scraping into it is bad
        """

        label = game_state.get("scene_label")
        rail_position = game_state.get("rail_position")

        if label not in ["half_wall_rail", "side_rail"]:
            return 0.0

        reward = 0.0

        if label == "half_wall_rail" or rail_position == "front":
            if action == "move_forward":
                reward += self.add_penalty("push_into_half_wall_rail", -15.0)
            elif action in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
                reward += 0.35
                self.reward_manager.add("half_wall_rail_escape_action", 0.35)

        elif label == "side_rail" or rail_position == "side":
            if action in ["strafe_left", "strafe_right"] and game_state.get("distance_moved", 0.0) <= 1.0:
                reward += self.add_penalty("scrape_side_rail", -8.0)
            elif action in ["move_forward", "turn_left", "turn_right"]:
                reward += 0.05
                self.reward_manager.add("side_rail_safe_action", 0.05)

        return reward

'''

if "def infer_half_wall_rail_state(" not in text:
    marker = "    def reset("
    if marker not in text:
        raise SystemExit("Could not find reset() marker")
    text = text.replace(marker, helper + "\n" + marker, 1)

# ---------------------------------------------------------
# 2. Patch sensory_scene_override to output half_wall_rail
# ---------------------------------------------------------
old_block = '''        # Correct fake-open predictions.
        if label == "open_path":
            if stuck_by_body or no_translation or near_spawn_boundary:
                label = "boundary_or_stuck_wall"
                confidence = max(confidence, 0.90)
            elif front_ratio >= 0.55:
                label = "front_wall"
                confidence = max(confidence, 0.75)
            elif side_ratio >= 0.85:
                label = "obstacle"
                confidence = max(confidence, 0.70)
'''

new_block = '''        # Correct fake-open predictions.
        if label in ["open_path", "unclear", "normal_navigation"]:
            # Half-walls / rails can look visually open above the rail,
            # but body feedback says the path is blocked.
            if front_ratio >= 0.52 and (stuck_by_body or no_translation):
                label = "half_wall_rail"
                confidence = max(confidence, 0.88)

            elif stuck_by_body or no_translation or near_spawn_boundary:
                label = "boundary_or_stuck_wall"
                confidence = max(confidence, 0.90)

            elif front_ratio >= 0.65:
                label = "front_wall"
                confidence = max(confidence, 0.80)

            elif front_ratio >= 0.52:
                label = "half_wall_rail"
                confidence = max(confidence, 0.75)

            elif side_ratio >= 0.85:
                label = "obstacle"
                confidence = max(confidence, 0.70)

            elif side_ratio >= 0.50:
                label = "side_rail"
                confidence = max(confidence, 0.65)
'''

if old_block in text:
    text = text.replace(old_block, new_block)
else:
    print("Warning: exact fake-open block not found. You may already changed it.")

# ---------------------------------------------------------
# 3. Add half-wall reward after scene_label is assigned post-action
# ---------------------------------------------------------
needle = '''        game_state["scene_label"] = scene_label
        game_state["scene_confidence"] = scene_confidence
        game_state["scene_probs"] = scene_probs
'''

insert = '''        rail_state = self.infer_half_wall_rail_state(
            scene_label=scene_label,
            scene_confidence=scene_confidence,
            wall_info=wall_info if "wall_info" in locals() else None,
            distance_moved=distance_moved if "distance_moved" in locals() else None,
            motion=motion if "motion" in locals() else None,
            action=action,
        )

        if rail_state.get("label") in ["half_wall_rail", "side_rail"]:
            scene_label = rail_state["label"]
            scene_confidence = rail_state["confidence"]

            game_state["scene_label"] = scene_label
            game_state["scene_confidence"] = scene_confidence
            game_state["rail_position"] = rail_state.get("position")
            game_state["rail_front_ratio"] = rail_state.get("front_ratio")
            game_state["rail_side_ratio"] = rail_state.get("side_ratio")

            if self._step_count % 25 == 0:
                print(
                    f"[half_wall_rail] label={scene_label} "
                    f"pos={rail_state.get('position')} "
                    f"front={rail_state.get('front_ratio'):.2f} "
                    f"side={rail_state.get('side_ratio'):.2f} "
                    f"action={action}"
                )

        reward += self.half_wall_rail_reward(game_state, action)
'''

if insert not in text:
    text = text.replace(needle, needle + "\n" + insert, 1)

# ---------------------------------------------------------
# 4. Include labels in obstacle checks
# ---------------------------------------------------------
text = text.replace(
    'scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall"]',
    'scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall", "half_wall_rail"]'
)

text = text.replace(
    'scene_label in ["front_wall", "door_or_button", "obstacle"]',
    'scene_label in ["front_wall", "door_or_button", "obstacle", "half_wall_rail"]'
)

# ---------------------------------------------------------
# 5. Compile-safe write
# ---------------------------------------------------------
p.write_text(text)
print("Added half_wall_rail / side_rail detection and reward.")
