from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_pre_action_reroute")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add init vars after step_count if missing.
marker = "        self.step_count = 0\n"
insert = """        # Limited director/rule-enforcer control for wall/rail rerouting.
        self.reroute_stuck_steps = 0
        self.reroute_override_steps = 0
        self.last_reroute_action = None
        self.last_reroute_reason = None
        self.previous_pre_action_position = None
"""

if "self.reroute_stuck_steps = 0" not in text:
    text = text.replace(marker, marker + insert, 1)

# Add reset vars after self.step_count = 0 in reset, if not already there.
reset_marker = "        self.step_count = 0\n        if hasattr(self, \"mission_tracker\"):"
reset_insert = """        self.step_count = 0
        self.reroute_stuck_steps = 0
        self.reroute_override_steps = 0
        self.last_reroute_action = None
        self.last_reroute_reason = None
        self.previous_pre_action_position = None
        if hasattr(self, "mission_tracker"):"""

if reset_marker in text:
    text = text.replace(reset_marker, reset_insert, 1)

helper = '''    def maybe_reroute_before_action(self, action_name, buttons, pre_game_state):
        """
        Limited control before make_action.

        This helps when the agent keeps pushing into rails/walls.
        It does not control normal movement.
        """

        debug = {
            "active": False,
            "reason": None,
            "stuck_steps": self.reroute_stuck_steps,
        }

        x = pre_game_state.get("x")
        y = pre_game_state.get("y")

        moved = 999.0
        if x is not None and y is not None:
            pos = (float(x), float(y))
            if self.previous_pre_action_position is not None:
                ox, oy = self.previous_pre_action_position
                moved = ((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2) ** 0.5
            self.previous_pre_action_position = pos

        pushing_action = action_name in ["move_forward", "strafe_left", "strafe_right"]

        if pushing_action and moved < 1.0:
            self.reroute_stuck_steps += 1
        else:
            self.reroute_stuck_steps = max(0, self.reroute_stuck_steps - 1)

        # Continue short reroute burst.
        if self.reroute_override_steps > 0:
            self.reroute_override_steps -= 1
            reroute_action = self.last_reroute_action or "turn_left"

            action_name = reroute_action
            buttons, action_name = self._action_to_buttons(
                self._action_name_to_index(action_name)
            )

            debug["active"] = True
            debug["reason"] = "continue_reroute"
            debug["stuck_steps"] = self.reroute_stuck_steps
            return action_name, buttons, debug

        # Start reroute only after repeated rail/wall pushing.
        if self.reroute_stuck_steps >= 6:
            if self.reroute_stuck_steps < 10:
                reroute_action = "move_backward"
            else:
                reroute_action = "turn_left"

            self.reroute_override_steps = 5
            self.last_reroute_action = reroute_action
            self.last_reroute_reason = "wall_or_rail_blocked"

            action_name = reroute_action
            buttons, action_name = self._action_to_buttons(
                self._action_name_to_index(action_name)
            )

            debug["active"] = True
            debug["reason"] = "wall_or_rail_blocked"
            debug["stuck_steps"] = self.reroute_stuck_steps

        return action_name, buttons, debug

'''

if "def maybe_reroute_before_action" not in text:
    marker2 = "    def _action_name_to_index(self, action_name):\n"
    if marker2 not in text:
        raise SystemExit("Could not find _action_name_to_index marker")
    text = text.replace(marker2, helper + "\n" + marker2, 1)

# Insert before make_action.
make_marker = "        try:\n            self.game.make_action(buttons, self.frame_skip)\n"

reroute_call = '''        action_name, buttons, reroute_debug = self.maybe_reroute_before_action(
            action_name=action_name,
            buttons=buttons,
            pre_game_state=pre_game_state,
        )

        if reroute_debug.get("active") and self.step_count % 25 == 0:
            print(
                f"[viz_reroute] reason={reroute_debug.get('reason')} "
                f"stuck_steps={reroute_debug.get('stuck_steps')} "
                f"action={action_name}"
            )

'''

if "reroute_debug = self.maybe_reroute_before_action" not in text:
    if make_marker not in text:
        raise SystemExit("Could not find make_action marker")
    text = text.replace(make_marker, reroute_call + make_marker, 1)

# Add info field.
text = text.replace(
    '"reward_debug": self.last_reward_debug,',
    '"reward_debug": self.last_reward_debug,\n            "reroute_debug": locals().get("reroute_debug", {}),',
)

p.write_text(text)
print("Added pre-action wall/rail reroute control.")
