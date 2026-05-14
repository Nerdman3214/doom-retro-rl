class CombatTactics:
    """
    Rule-based tactical layer.

    This does not replace PPO.
    It corrects obviously bad combat behavior:
    - don't stare at walls
    - dodge when hurt
    - prioritize close enemies
    - shoot only when aligned
    - move when no target exists
    """

    def choose_combat_action(self, state):
        enemy_visible = state.get("enemy_visible", False)
        enemy_centered = state.get("enemy_centered", False)
        enemy_left = state.get("enemy_left", False)
        enemy_right = state.get("enemy_right", False)
        enemy_close = state.get("enemy_close", False)
        health = state.get("health", 100)
        ammo = state.get("ammo", 0)
        incoming_damage = state.get("health_delta", 0) < 0

        if not enemy_visible:
            return None

        # If taking damage, dodge first unless already lined up.
        if incoming_damage and not enemy_centered:
            if enemy_left:
                return "strafe_right"
            if enemy_right:
                return "strafe_left"
            return "strafe_left"

        # Dangerous close enemy: back up while fighting.
        if enemy_close and health < 50:
            if enemy_centered and ammo > 0:
                return "shoot"
            return "move_backward"

        # Aim shortest direction.
        if enemy_left:
            return "turn_left"

        if enemy_right:
            return "turn_right"

        # If centered, attack.
        if enemy_centered:
            if ammo > 0:
                return "shoot"
            return "melee_attack"

        return None