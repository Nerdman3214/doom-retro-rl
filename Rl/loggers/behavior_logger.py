import os
import numpy as np


class BehaviorLogger:

    def __init__(self):

        self.dataset_path = "dataset"

        os.makedirs(self.dataset_path, exist_ok=True)

        self.index = 0


    def log(self, frame, action, health, ammo):

        filename = f"{self.dataset_path}/sample_{self.index}.npz"


        np.savez(
            filename,
            frame=frame,
            action=action,
            health=health,
            ammo=ammo
        )


        self.index += 1