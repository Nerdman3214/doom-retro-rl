POSITIVE_REWARD_TEST_MULTIPLIER = 3.0


def positive_reward(value):
    try:
        value = float(value)
    except Exception:
        return value

    if value > 0:
        return value * POSITIVE_REWARD_TEST_MULTIPLIER

    return value


def compute_doom_brain_reward(game_state, action_name, world_state, mission_update):
    """
    Shared reward logic for DoomEnv and VizDoomEnv.

    This is intentionally simple and stable.
    It does not assume one backend.
    """

    reward = 0.0
    events = []

    mode = world_state.get("mode", "navigate")
    health = int(game_state.get("health", 100) or 100)
    ammo = int(game_state.get("ammo", 0) or 0)

    enemy_visible = bool(game_state.get("enemy_visible", False))
    enemy_centered = bool(game_state.get("enemy_centered", False))
    front_blocked = bool(game_state.get("front_blocked", False))

    damage_taken = float(game_state.get("damage_taken", 0.0) or 0.0)
    hit_enemy = bool(game_state.get("hit_enemy", False))
    killed_enemy = bool(game_state.get("killed_enemy", False))
    picked_item = bool(game_state.get("picked_item", False))
    picked_health = bool(game_state.get("picked_health", False))
    picked_ammo = bool(game_state.get("picked_ammo", False))
    picked_weapon = bool(game_state.get("picked_weapon", False))
    opened_door = bool(game_state.get("opened_door", False))

    repeated_position_steps = int(game_state.get("repeated_position_steps", 0) or 0)

    # Mission navigation / secrets / exit.
    mission_reward = float(mission_update.get("reward", 0.0) or 0.0)

    if mode == "navigate":
        reward += mission_reward * 1.5
    elif mode == "fight":
        reward += mission_reward * 0.4
    elif mode == "unstuck":
        reward += mission_reward * 0.5
    elif mode in {"collect_health", "collect_ammo"}:
        reward += mission_reward * 0.7
    else:
        reward += mission_reward

    for ev in mission_update.get("events", []):
        events.append(ev)

    # Combat.
    if enemy_visible:
        if action_name == "shoot" and ammo > 0:
            reward += positive_reward(15.0)
            events.append("shoot_visible_enemy")

        if enemy_centered and action_name == "shoot" and ammo > 0:
            reward += positive_reward(15.0)
            events.append("shoot_centered_enemy")

        if action_name in {"move_forward"} and damage_taken > 0:
            reward -= 15.0
            events.append("pushed_into_damage")

    if hit_enemy:
        reward += positive_reward(0.75)
        events.append("hit_enemy")

    if killed_enemy:
        reward += positive_reward(1.50)
        events.append("killed_enemy")

    if action_name == "shoot" and not enemy_visible:
        reward -= 15.0
        events.append("blind_shot")

    # Survival.
    if damage_taken > 0:
        reward -= 15.0
        events.append("damage_taken")

    if health <= 25:
        reward -= 15.0
        events.append("critical_health")

    # Items/resources.
    if picked_item:
        reward += positive_reward(15.0)
        events.append("picked_item")

    if picked_health:
        reward += positive_reward(15.0)
        events.append("picked_health")

    if picked_ammo:
        reward += positive_reward(15.0)
        events.append("picked_ammo")

    if picked_weapon:
        reward += positive_reward(1.00)
        events.append("picked_weapon")

    if ammo <= 5 and action_name == "shoot":
        reward -= 15.0
        events.append("wasted_low_ammo")

    # Door/use logic.
    if opened_door:
        reward += positive_reward(15.0)
        events.append("opened_door")

    if action_name == "use":
        if front_blocked:
            reward += positive_reward(0.10)
            events.append("use_near_blocker")
        else:
            reward -= 15.0
            events.append("use_spam_or_unclear")

    # Unstuck / wall logic.
    if mode == "unstuck":
        if action_name in {"turn_left", "turn_right", "move_backward", "strafe_left", "strafe_right"}:
            reward += positive_reward(15.0)
            events.append("unstuck_action")
        if action_name == "move_forward" and front_blocked:
            reward -= 15.0
            events.append("pushing_wall")

    if repeated_position_steps >= 10:
        reward -= 15.0
        events.append("repeated_position")

    # Small living cost.
    reward -= 0.001

    return float(reward), events
