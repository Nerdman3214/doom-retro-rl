class HelperBot:
    def get_action(self, game_state):
        health = int(game_state.get("health", 100) or 100)
        ammo = int(game_state.get("ammo", 0) or 0)
        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        enemy_left = bool(game_state.get("enemy_left", False))
        enemy_right = bool(game_state.get("enemy_right", False))
        stage = int(game_state.get("curriculum_stage", 0) or 0)

        near_use_point = bool(
            game_state.get("near_use_point", False)
            or game_state.get("door_visible", False)
            or game_state.get("door_centered", False)
            or game_state.get("director_hint_action") == "use"
        )

        health_pickup_visible = bool(game_state.get("health_pickup_visible", False))
        ammo_pickup_visible = bool(game_state.get("ammo_pickup_visible", False))

        front_blocked = bool(
            game_state.get("front_blocked", False)
            or game_state.get("front_wall_close", False)
            or float(game_state.get("front_wall_ratio", 0.0) or 0.0) >= 0.45
        )

        if enemy_visible:
            if enemy_centered and ammo > 0:
                return "shoot"

            if enemy_left:
                return "turn_left"

            if enemy_right:
                return "turn_right"

            if ammo <= 0:
                return "move_backward"

            return "turn_right"

        if health <= 25 and health_pickup_visible:
            return "move_forward"

        if ammo <= 0 and ammo_pickup_visible:
            return "move_forward"

        if near_use_point:
            return "use"

        if front_blocked:
            return "turn_right"

        if stage >= 2:
            return "move_forward"

        return "move_forward"