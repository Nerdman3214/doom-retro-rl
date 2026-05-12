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
        self.previous_ammo = None
        self.previous_weapon = None
        self.previous_kill_count = None
        self.previous_health = None

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

    def _estimate_current_weapon(self, ammo, health=None):
        """
        Best-effort weapon estimate.

        Since Doom Retro/Freedoom does not automatically give us a clean
        weapon name through normal screen capture, this starts as a safe
        placeholder.

        Later, this can be upgraded using:
        - shared memory
        - HUD weapon detection
        - keypress tracking
        - OCR/HUD image classification
        """

        # If ammo is 0, the player may be on fist/chainsaw/ripter,
        # but this is not guaranteed.
        if ammo is not None and ammo <= 0:
            return "melee_unknown"

        # Default assumption at pistol start.
        return "pistol"
    def _estimate_kill_delta(self, health_delta, enemy_visible):
        """
        Best-effort kill estimate.

        This is intentionally conservative. Without direct game stats,
        we do not truly know when an enemy died.

        Later, improve this with:
        - enemy disappearance after centered shots
        - corpse detection
        - dropped ammo pickup
        - shared memory kill count
        """

        # Placeholder: do not guess kills yet.
        return 0

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

        health = 100
        ammo = 50
        armor = 0
        x = None
        y = None
        shared_state_available = False

        # -----------------------------------------------------
        # Delta tracking
        # -----------------------------------------------------

        health_delta = 0
        ammo_delta = 0
        weapon_delta = 0
        kill_delta = 0

        if self.previous_health is not None and health is not None:
            health_delta = health - self.previous_health

        if self.previous_ammo is not None and ammo is not None:
            ammo_delta = ammo - self.previous_ammo

        current_weapon = self._estimate_current_weapon(ammo, health)

        if self.previous_weapon is not None and current_weapon != self.previous_weapon:
            weapon_delta = 1

        kill_delta = self._estimate_kill_delta(
        health_delta=health_delta,
        enemy_visible=False,
    )

        enemy_visible = False

        try:
            enemy_visible = bool(game_state.get("enemy_visible", False))
        except Exception:
            enemy_visible = False

        kill_delta = self._estimate_kill_delta(health_delta, enemy_visible)

        self.previous_health = health
        self.previous_ammo = ammo
        self.previous_weapon = current_weapon
        return state{
            "health": health,
            "ammo": ammo,
            "armor": armor,

            "health_delta": health_delta,
            "ammo_delta": ammo_delta,

            "weapon": current_weapon,
            "weapon_delta": weapon_delta,
            "kill_delta": kill_delta,

            "x": x,
            "y": y,
            "shared_state_available": shared_state_available,
        }

    def reset_tracking(self):
        self.previous_ammo = None
        self.previous_weapon = None
        self.previous_kill_count = None
        self.previous_health = None
    


    def get_frame(self):
        raw, w, h = frame_cache.get_raw()
        resized = frame_processor.resize_bgra_to_bgr(raw, w, h, OUT_W, OUT_H)
        return np.frombuffer(resized, dtype=np.uint8).reshape(OUT_H, OUT_W, 3).copy()
    

    def build(self):
        return self.get_frame()