import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import deque
from utils.shm_reader import read_player_state
import frame_cache
import frame_processor

OUT_W, OUT_H = 84, 84


class ObservationBuilder:

    def __init__(self):

        self.frames = deque(maxlen=4)

        # Game state tracking from shared memory if DOOM Retro is running with the RL hook.
        self.health = 100
        self.ammo = 50
        self.prev_health = 100
        self.prev_ammo = 50
        self.x = None
        self.y = None
        self.angle = None
        self.armor = None
        self.kills = None
        self.shared_state_available = False

    def get_game_state(self):
        """Returns a dict of tracked game variables and their deltas."""
        shared = read_player_state()
        if shared is not None:
            self.shared_state_available = True
            self.health = shared["health"]
            self.ammo = sum(shared["ammo"])
            self.x = shared["x"]
            self.y = shared["y"]
            self.angle = shared["angle"]
            self.armor = shared["armor"]
            self.kills = shared["kills"]

        state = {
            "health": self.health,
            "ammo": self.ammo,
            "health_delta": self.health - self.prev_health,
            "ammo_delta": self.ammo - self.prev_ammo,
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "armor": self.armor,
            "kills": self.kills,
            "shared_state_available": self.shared_state_available,
        }
        self.prev_health = self.health
        self.prev_ammo = self.ammo
        return state
    


    def get_frame(self):
        raw, w, h = frame_cache.get_raw()
        resized = frame_processor.resize_bgra_to_bgr(raw, w, h, OUT_W, OUT_H)
        return np.frombuffer(resized, dtype=np.uint8).reshape(OUT_H, OUT_W, 3).copy()


    def build(self):

        frame = self.get_frame()

        self.frames.append(frame)

        while len(self.frames) < 4:

            self.frames.append(frame)

        stacked = np.concatenate(list(self.frames), axis=2)

        return stacked