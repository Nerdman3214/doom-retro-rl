class ActionSpace:

    ACTIONS = [
        "move_forward",
        "turn_left",
        "turn_right",
        "turn_left_shoot",
        "turn_right_shoot",
        "move_forward_shoot",
        "shoot",
        "Strafe_Left",
        "Strafe_Right",
        "move_backward",
        "move_backward_shoot",
        "swap_weapon",
        "use"
        ] 

    def sample(self):
        import random
        return random.choice(self.ACTIONS)