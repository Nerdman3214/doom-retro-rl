def choose_world_mode(game_state, sensory_state=None):
    """
    Converts raw game/sensory data into a high-level mode.
    This should classify bad reset/stuck states before normal navigation.
    """

    sensory_state = sensory_state or {}

    health = int(game_state.get("health", 100) or 100)
    ammo = int(game_state.get("ammo", 0) or 0)

    x = game_state.get("x")
    y = game_state.get("y")

    enemy_visible = bool(game_state.get("enemy_visible", False))
    enemy_centered = bool(game_state.get("enemy_centered", False))
    front_blocked = bool(game_state.get("front_blocked", False))

    situation = sensory_state.get(
        "situation",
        game_state.get("sensory_situation", "normal_navigation"),
    )

    repeated_steps = int(game_state.get("repeated_position_steps", 0) or 0)
    no_position_steps = int(game_state.get("no_position_change_steps", 0) or 0)

    # Real Freedoom E1M1 start is about (-416, 256).
    # If we are very far from that immediately after reset, this is not normal navigation.
    if x is not None and y is not None:
        x = float(x)
        y = float(y)

        if x > 300 and y < 600:
            return "wrong_route_or_bad_reset"

    if repeated_steps >= 8 or no_position_steps >= 5:
        return "unstuck"

    if situation in ["stuck_or_looping", "spawn_wall_zone", "front_blocked"]:
        return "unstuck"

    if situation in ["right_route_area", "secret_side_area"]:
        return "wrong_route_or_side_area"

    if health <= 35:
        return "survive"

    if enemy_visible or enemy_centered:
        if ammo > 0:
            return "fight"
        return "evade"

    if front_blocked:
        return "unstuck"

    return "navigate"