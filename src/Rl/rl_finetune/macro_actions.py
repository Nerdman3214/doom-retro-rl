from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroAction:
    name: str
    force_buttons: tuple[str, ...] = ()
    block_buttons: tuple[str, ...] = ()
    mouse_dx: float | None = None
    tap_use: bool = False


MACRO_ACTIONS: list[MacroAction] = [
    MacroAction(
        name="trust_bc",
    ),
    MacroAction(
        name="force_forward",
        force_buttons=("move_forward",),
        block_buttons=("move_backward",),
    ),
    MacroAction(
        name="forward_strafe_left",
        force_buttons=("move_forward", "strafe_left"),
        block_buttons=("move_backward", "strafe_right"),
    ),
    MacroAction(
        name="forward_strafe_right",
        force_buttons=("move_forward", "strafe_right"),
        block_buttons=("move_backward", "strafe_left"),
    ),
    MacroAction(
        name="forward_turn_left",
        force_buttons=("move_forward",),
        block_buttons=("move_backward",),
        mouse_dx=-1.0,
    ),
    MacroAction(
        name="forward_turn_right",
        force_buttons=("move_forward",),
        block_buttons=("move_backward",),
        mouse_dx=1.0,
    ),
    MacroAction(
        name="tap_use_forward",
        force_buttons=("move_forward", "use"),
        block_buttons=("move_backward",),
        tap_use=True,
    ),
    MacroAction(
        name="combat_forward",
        force_buttons=("move_forward", "mouse_left"),
        block_buttons=("move_backward",),
    ),
]


def apply_macro(
    active: set[str],
    dx: float,
    macro_id: int,
) -> tuple[set[str], float, str]:
    """
    Apply high-level RL macro action on top of BC imitation output.

    active:
        Current BC-predicted active buttons.

    dx:
        Current BC-predicted horizontal mouse movement.

    macro_id:
        Index into MACRO_ACTIONS.

    returns:
        new_active, new_dx, macro_name
    """
    macro = MACRO_ACTIONS[int(macro_id)]
    active = set(active)

    for b in macro.block_buttons:
        active.discard(b)

    for b in macro.force_buttons:
        active.add(b)

    if macro.mouse_dx is not None:
        dx = float(macro.mouse_dx)

    return active, dx, macro.name


if __name__ == "__main__":
    for i, macro in enumerate(MACRO_ACTIONS):
        print(i, macro)
