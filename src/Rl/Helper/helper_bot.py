class HelperBot:

    def get_action(self, game_state):
        health = game_state.get("health", 100)
        ammo = game_state.get("ammo", 0)
        enemy_visible = game_state.get("enemy_visible", False)
        enemy_centered = game_state.get("enemy_centered", False)
        stage = game_state.get("curriculum_stage", 0)

        if health < 30:
            return "move_backward"

        if enemy_visible:
            if enemy_centered and ammo > 0:
                return "shoot"
            return "turn_right"

        if stage == 3:
            # Cautious combat search: keep progressing, but don't spam shoot.
            if game_state.get("near_door", False):
                return "use"
            return "move_forward"

        return "move_forward"