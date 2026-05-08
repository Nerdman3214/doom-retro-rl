import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.shm_reader import read_player_state
import frame_cache
import frame_processor

OUT_W, OUT_H = 84, 84


class ObservationBuilder:

    def __init__(self):

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
        self.goal_position = None
        self.estimated_goal = None

    def get_player_position(self):
        """Return player position (x, y) from shared memory or fallback."""
        shared = read_player_state()
        if shared is not None and self.x is not None:
            return (self.x, self.y)
        # Fallback: try to get from screen state or return center
        return (self.x if self.x else 0.0, self.y if self.y else 0.0)

    def get_goal_position(self):
        """
        Return estimated goal position (x, y).
        
        On Doom maps, the exit/goal is typically in a known corner.
        For now, return a heuristic estimate or None if not determinable.
        A more robust approach would require map metadata or exit detection.
        """
        if self.goal_position is not None:
            return self.goal_position
        # Heuristic: assume goal is somewhere else on the map
        # This should be replaced with actual goal detection if available
        if self.x is not None and self.y is not None:
            # For E1M1, goal is roughly opposite corner
            # Use estimated position based on current position
            self.estimated_goal = (self.x + 1000, self.y + 1000)
            return self.estimated_goal
        return (0.0, 0.0)

    def set_goal_position(self, x, y):
        """Manually set the goal position (called if known from map data)."""
        self.goal_position = (x, y)

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
        return self.get_frame()