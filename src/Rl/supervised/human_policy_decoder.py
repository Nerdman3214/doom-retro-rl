ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]


DEFAULT_THRESHOLDS = {
    "move_forward": 0.55,
    "move_backward": 0.80,
    "turn_left": 0.45,
    "turn_right": 0.60,
    "strafe_left": 0.60,
    "strafe_right": 0.55,
    "shoot": 0.65,
    "use": 0.55,
}


def load_thresholds(path=None):
    thresholds = dict(DEFAULT_THRESHOLDS)

    if path is None:
        return thresholds

    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return thresholds

    for line in p.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        thresholds[key.strip()] = float(value.strip())

    return thresholds


def decode_teacher_buttons(probs, thresholds=None, max_buttons=3):
    """
    Convert raw sigmoid probabilities into a safe Doom button vector.

    This avoids impossible or messy outputs like:
      move_forward + move_backward
      turn_left + turn_right
      strafe_left + strafe_right
      six movement buttons at once
    """

    thresholds = thresholds or DEFAULT_THRESHOLDS

    prob_map = {
        ACTION_NAMES[i]: float(probs[i])
        for i in range(len(ACTION_NAMES))
    }

    chosen = set()

    # 1. High-confidence special actions.
    # Shoot should be allowed with movement, but only if clearly confident.
    if prob_map["shoot"] >= thresholds["shoot"]:
        chosen.add("shoot")

    # Use is usually context-specific; only allow it when confident.
    if prob_map["use"] >= thresholds["use"]:
        chosen.add("use")

    # 2. Pick best forward/backward movement.
    fb_candidates = ["move_forward", "move_backward"]
    best_fb = max(fb_candidates, key=lambda k: prob_map[k])
    if prob_map[best_fb] >= thresholds[best_fb]:
        chosen.add(best_fb)

    # 3. Pick best turning action.
    turn_candidates = ["turn_left", "turn_right"]
    best_turn = max(turn_candidates, key=lambda k: prob_map[k])
    if prob_map[best_turn] >= thresholds[best_turn]:
        chosen.add(best_turn)

    # 4. Pick best strafing action.
    strafe_candidates = ["strafe_left", "strafe_right"]
    best_strafe = max(strafe_candidates, key=lambda k: prob_map[k])
    if prob_map[best_strafe] >= thresholds[best_strafe]:
        chosen.add(best_strafe)

    # 5. Avoid too much lateral + turning noise.
    # Prefer turning over strafing unless strafe is much stronger.
    if best_turn in chosen and best_strafe in chosen:
        if prob_map[best_strafe] < prob_map[best_turn] + 0.10:
            chosen.discard(best_strafe)

    # 6. Keep max buttons.
    # Priority: shoot/use first, then movement.
    priority = [
        "shoot",
        "use",
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
    ]

    ordered = [a for a in priority if a in chosen]
    ordered = ordered[:max_buttons]

    buttons = [0] * len(ACTION_NAMES)
    for action in ordered:
        buttons[ACTION_NAMES.index(action)] = 1

    label = "+".join(ordered) if ordered else "no_op"

    return buttons, label, prob_map
