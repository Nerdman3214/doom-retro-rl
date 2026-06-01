def choose_world_mode(game_state, sensory_state=None):
    """
    High-level Doom mode selector.

    This is not the policy.
    This tells reward/director logic what matters most right now.
    """

    sensory_state = sensory_state or {}

    situation = sensory_state.get(
        "situation",
        game_state.get("sensory_situation", "normal_navigation"),
    )

    health = int(game_state.get("health", 100) or 100)
    ammo = int(game_state.get("ammo", 0) or 0)

    enemy_visible = bool(game_state.get("enemy_visible", False))
    enemy_centered = bool(game_state.get("enemy_centered", False))
    front_blocked = bool(game_state.get("front_blocked", False))

    repeated_position_steps = int(game_state.get("repeated_position_steps", 0) or 0)
    no_position_change_steps = int(game_state.get("no_position_change_steps", 0) or 0)

    # Highest priority: danger and broken movement.
    if health <= 25:
        return "survive"

    if situation in {
        "stuck_or_looping",
        "spawn_wall_zone",
        "front_blocked",
        "secret_side_area",
        "right_route_area",
    }:
        return "unstuck"

    if front_blocked:
        return "unstuck"

    if repeated_position_steps >= 8 or no_position_change_steps >= 5:
        return "unstuck"

    # Combat before route pushing.
    if enemy_visible or enemy_centered:
        if ammo > 0:
            return "fight"
        return "evade"

    # Resource mode.
    if health <= 45:
        return "collect_health"

    if ammo <= 5:
        return "collect_ammo"

    return "navigate"


def build_world_state(game_state, sensory_state=None):
    sensory_state = sensory_state or {}

    mode = choose_world_mode(game_state, sensory_state)

    return {
        "mode": mode,
        "situation": sensory_state.get(
            "situation",
            game_state.get("sensory_situation", "normal_navigation"),
        ),
        "enemy_visible": bool(game_state.get("enemy_visible", False)),
        "enemy_centered": bool(game_state.get("enemy_centered", False)),
        "front_blocked": bool(game_state.get("front_blocked", False)),
        "low_health": int(game_state.get("health", 100) or 100) <= 45,
        "low_ammo": int(game_state.get("ammo", 0) or 0) <= 5,
        "health": int(game_state.get("health", 100) or 100),
        "ammo": int(game_state.get("ammo", 0) or 0),
        "x": game_state.get("x"),
        "y": game_state.get("y"),
    }
