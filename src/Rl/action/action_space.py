class ActionSpace:
    ACTIONS = [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "shoot",
        "use",
        "swap_weapon",
        "melee_attack"
    ]

    def sample(self):
        import random
        return random.choice(self.ACTIONS)