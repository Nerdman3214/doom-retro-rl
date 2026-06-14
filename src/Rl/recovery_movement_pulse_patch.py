from pathlib import Path
import re

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_recovery_movement_pulse_patch")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ---------------------------------------------------------
# 1. Add recovery memory fields in __init__
# ---------------------------------------------------------

init_marker = '''        self.recent_loop_steps = 0
'''

init_insert = '''        self.recovery_pulse_steps = 0
        self.recovery_escape_side = "right"
        self.recovery_last_position = None
        self.recovery_no_move_steps = 0
'''

if init_insert not in text:
    if init_marker not in text:
        print("Warning: init marker not found. Recovery fields may already exist or marker changed.")
    else:
        text = text.replace(init_marker, init_marker + init_insert, 1)

# ---------------------------------------------------------
# 2. Replace directional_recovery_action safely
# ---------------------------------------------------------

new_function = '''    def directional_recovery_action(
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
        Combined sensory + vision recovery with real movement.

        Recovery should create translation, not just rotation:

            turn away -> move backward -> strafe away -> test forward
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
            or front >= 0.42
            or left >= 0.45
            or right >= 0.45
            or self.stuck_counter >= 2
            or self.wall_contact_steps >= 1
            or low_translation
            or recommended in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        )

        if not recovery_needed:
            self.recovery_pulse_steps = 0
            return action

        # Choose escape side based on obstacle pressure.
        if left > right + 0.05:
            escape_side = "right"
        elif right > left + 0.05:
            escape_side = "left"
        else:
            escape_side = getattr(self, "recovery_escape_side", "right")

        self.recovery_escape_side = escape_side

        turn_away = "turn_right" if escape_side == "right" else "turn_left"
        strafe_away = "strafe_right" if escape_side == "right" else "strafe_left"

        front_blocked = front >= 0.42 or situation == "front_blocked"

        self.recovery_pulse_steps += 1
        phase = self.recovery_pulse_steps % 8

        # Boxed or front-blocked: first rotate, then actually move away.
        if front_blocked:
            if phase in [1, 2]:
                return turn_away
            if phase in [3, 4]:
                return "move_backward"
            if phase in [5, 6]:
                return strafe_away
            return "move_forward"

        # Side obstacle: move away from that side.
        if left >= 0.45 or right >= 0.45:
            if phase in [1, 2]:
                return strafe_away
            if phase in [3, 4]:
                return turn_away
            if phase in [5, 6]:
                return "move_forward"
            return "move_backward"

        # Stuck loop without obvious wall: force translation.
        if situation == "stuck_or_looping" or self.stuck_counter >= 2:
            if phase in [1, 2]:
                return "move_backward"
            if phase in [3, 4]:
                return turn_away
            if phase in [5, 6]:
                return strafe_away
            return "move_forward"

        return action

'''

pattern = r'''    def directional_recovery_action\(
        self,
        action,
        wall_info=None,
        sensory_state=None,
        scene_label=None,
        scene_confidence=0\.0,
        distance_moved=None,
        motion=None,
    \):
.*?(?=\n    def )'''

text, count = re.subn(pattern, new_function.rstrip(), text, flags=re.DOTALL)

if count == 0:
    marker = "    def reset("
    if marker not in text:
        raise SystemExit("Could not find directional_recovery_action or reset() marker.")
    text = text.replace(marker, new_function + "\n" + marker, 1)
    print("Inserted new directional_recovery_action before reset().")
else:
    print(f"Replaced directional_recovery_action. count={count}")

p.write_text(text)
print("Applied recovery movement pulse patch.")
