from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_reroute_helper")
backup.write_text(text)
print(f"Backup saved to {backup}")

helper = '''    def maybe_reroute_from_blocked_path(
        self,
        action_name,
        buttons,
        pre_game_state,
        director_result=None,
    ):
        """
        Limited director/rule-enforcer style override.

        Purpose:
        - If the agent keeps pushing into rails/walls or makes no exit progress,
          briefly redirect it.
        - This should only activate when stuck, not control the agent every step.

        Returns:
            action_name, buttons, debug_info
        """

        debug = {
            "active": False,
            "reason": None,
            "stuck_steps": self.reroute_stuck_steps,
        }

        distance_delta = 0.0
        if director_result is not None:
            distance_delta = float(director_result.get("distance_delta", 0.0) or 0.0)

        # Estimate blocked behavior.
        distance_moved = float(pre_game_state.get("distance_moved", 0.0) or 0.0)
        pushing_action = action_name in ["move_forward", "strafe_left", "strafe_right"]

        no_exit_progress = distance_delta < 1.0
        barely_moving = distance_moved < 2.0

        if pushing_action and no_exit_progress and barely_moving:
            self.reroute_stuck_steps += 1
        else:
            self.reroute_stuck_steps = max(0, self.reroute_stuck_steps - 1)

        # Continue short override burst if already active.
        if self.reroute_override_steps > 0:
            self.reroute_override_steps -= 1

            if self.last_reroute_action in self.ACTIONS:
                action_name = self.last_reroute_action
                buttons, action_name = self._action_to_buttons(
                    self._action_name_to_index(action_name)
                )

                debug["active"] = True
                debug["reason"] = "continuing_reroute"
                debug["stuck_steps"] = self.reroute_stuck_steps
                return action_name, buttons, debug

        # Start a new reroute burst.
        if self.reroute_stuck_steps >= 8:
            # Simple escape pattern:
            # first back up, then turn left to find the real path around rails/walls.
            if self.reroute_stuck_steps < 12:
                reroute_action = "move_backward"
            else:
                reroute_action = "turn_left"

            self.reroute_override_steps = 4
            self.last_reroute_action = reroute_action
            self.last_reroute_reason = "blocked_path_or_rail"

            action_name = reroute_action
            buttons, action_name = self._action_to_buttons(
                self._action_name_to_index(action_name)
            )

            debug["active"] = True
            debug["reason"] = "blocked_path_or_rail"
            debug["stuck_steps"] = self.reroute_stuck_steps

        return action_name, buttons, debug

'''

if "def maybe_reroute_from_blocked_path" not in text:
    marker = "    def _action_name_to_index(self, action_name):\n"
    if marker not in text:
        raise SystemExit("Could not find _action_name_to_index marker")

    text = text.replace(marker, helper + "\n" + marker, 1)

p.write_text(text)
print("Added maybe_reroute_from_blocked_path helper.")
