class HelperBot:

    def get_action(self, game_state):


        health = game_state["health"]
        enemy_visible = game_state.get("enemy_visible", False)

        # PRIORITY 1: survival
        if health < 30:
            return "move_backward"

        # PRIORITY 2: combat
        if enemy_visible:
            return "shoot"

        # PRIORITY 3: exploration
        return "move_forward"