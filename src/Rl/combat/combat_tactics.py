class CombatTactics:
    """
    Conservative combat helper.

    This should assist PPO, not hijack the whole policy.
    It only returns strong combat actions when the situation is clear.
    """

    def __init__(self):
        self.no_ammo_swap_cooldown = 0

    def choose_combat_action(self, state):
        stage = state.get("curriculum_stage", 0)

        enemy_visible = state.get("enemy_visible", False)
        enemy_centered = state.get("enemy_centered", False)
        enemy_left = state.get("enemy_left", False)
        enemy_right = state.get("enemy_right", False)
        enemy_close = state.get("enemy_close", False)

        health = state.get("health", 100)
        ammo = state.get("ammo", 0)
        health_delta = state.get("health_delta", 0)

        action = state.get("action", None)

        # No combat helper before combat stages.
        if stage < 3:
            return None

        # If no enemy, do not interfere with exploration, doors, or movement.
        if not enemy_visible:
            return None

        # Let use/door actions survive in route stages.
        if stage >= 5 and action == "use":
            return None

        # If stuck or route progress is needed, do not hijack movement forever.
        stuck_counter = state.get("stuck_counter", 0)
        distance_traveled = state.get("distance_traveled", 0.0)

        if stage >= 5 and distance_traveled < 50.0:
            if action in ["move_forward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
                return None

        if stuck_counter >= 6:
            if action in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
                return None

        # No ammo behavior:
        # Do NOT spam swap_weapon forever.
        if ammo <= 0:
            if self.no_ammo_swap_cooldown > 0:
                self.no_ammo_swap_cooldown -= 1

                # If enemy is not centered, turn toward it instead of swapping.
                if enemy_left:
                    return "turn_left"
                if enemy_right:
                    return "turn_right"

                # If enemy is centered and close, allow melee.
                if enemy_centered and enemy_close:
                    return "melee_attack"

                # Otherwise move to avoid standing still.
                return "move_backward"

            # Try swap occasionally, not every frame.
            self.no_ammo_swap_cooldown = 12
            return "swap_weapon"

        # If taking damage, dodge.
        if health_delta < 0:
            if enemy_left:
                return "strafe_right"
            if enemy_right:
                return "strafe_left"
            return "strafe_left"

        # If low health and enemy is close, retreat.
        if health < 45 and enemy_close:
            return "move_backward"

        # Aim first. This fixes your "shoots but does not rotate toward enemy" issue.
        if enemy_left:
            return "turn_left"

        if enemy_right:
            return "turn_right"

        # Shoot only when centered.
        if enemy_centered and ammo > 0:
            return "shoot"

        return None