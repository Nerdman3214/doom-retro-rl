#!/usr/bin/env python3
from pathlib import Path

p = Path("tools/watch_vizdoom_z_breadcrumb_agent.py")
s = p.read_text()
backup = p.with_suffix(".py.before_bc_route_only_line_patch")
backup.write_text(s)
print("[patch] backup saved:", backup)

new_source_safe = '''def bc_source_is_safe(source):
    """Only allow BC on plain route-following.

    The BC student is still learning movement. Keep all door/use/final/
    terminal/combat/recovery logic under the scripted teacher.
    """
    src = str(source).lower()

    allowed_prefixes = (
        "ghost_route@",
        "route_action",
        "z_steer",
        "noop_to_steer",
    )

    if not src.startswith(allowed_prefixes):
        return False

    blocked = [
        "door",
        "goal",
        "final",
        "terminal",
        "combat",
        "damage",
        "stuck",
        "recover",
        "recovery",
        "use",
        "shoot",
        "magnet",
        "rescue",
        "wall",
        "after_use",
        "cross",
    ]

    return not any(tok in src for tok in blocked)
'''

new_names_to_buttons = '''def bc_names_to_buttons(action_names, probs, threshold):
    """Convert BC probabilities to a conservative route-movement action.

    Early BC is not allowed to press use/shoot/back/strafe. This prevents
    hard flicks and bad final-door behavior while we are still testing.
    """
    raw_names = []
    for name, prob in zip(action_names, probs):
        if float(prob) >= threshold:
            raw_names.append(name)

    forbidden = {
        "use",
        "shoot",
        "move_backward",
        "strafe_left",
        "strafe_right",
    }

    if any(name in forbidden for name in raw_names):
        return [], button_vec()

    if "turn_left" in raw_names and "turn_right" in raw_names:
        return [], button_vec()

    if "move_forward" not in raw_names:
        return [], button_vec()

    names = [
        name for name in raw_names
        if name in ("move_forward", "turn_left", "turn_right")
    ]

    return names, button_vec(*names)
'''


def replace_top_level_func(text, func_name, replacement):
    lines = text.splitlines(keepends=True)
    start = None

    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
            break

    if start is None:
        return text, False

    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line.startswith("def ") or line.startswith("class "):
            end = j
            break

    new_lines = lines[:start] + [replacement.rstrip() + "\n\n"] + lines[end:]
    return "".join(new_lines), True


s, ok1 = replace_top_level_func(s, "bc_source_is_safe", new_source_safe)
s, ok2 = replace_top_level_func(s, "bc_names_to_buttons", new_names_to_buttons)

if not ok1:
    raise SystemExit("[patch] could not find bc_source_is_safe")
if not ok2:
    raise SystemExit("[patch] could not find bc_names_to_buttons")

p.write_text(s)
print("[patch] replaced bc_source_is_safe:", ok1)
print("[patch] replaced bc_names_to_buttons:", ok2)
