import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from utils.shm_reader import read_player_state
import frame_cache

import cv2


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
        self.armor = 0
        self.kills = 0

        self.prev_kills = 0
        self.previous_weapon = None

        self.shared_state_available = False
        self.goal_position = None
        self.estimated_goal = None

    # ---------------------------------------------------------
    # Reset helpers
    # ---------------------------------------------------------

    def reset_tracking(self):
        self.prev_health = self.health
        self.prev_ammo = self.ammo
        self.prev_kills = self.kills if self.kills is not None else 0
        self.previous_weapon = None

    # ---------------------------------------------------------
    # Position / goal helpers
    # ---------------------------------------------------------

    def get_player_position(self):
        """Return player position (x, y) from shared memory or fallback."""
        shared = read_player_state()

        if shared is not None:
            self.shared_state_available = True
            self.x = shared.get("x", self.x)
            self.y = shared.get("y", self.y)

        return (
            self.x if self.x is not None else 0.0,
            self.y if self.y is not None else 0.0,
        )

    def get_goal_position(self):
        """
        Return estimated goal position (x, y).

        This is still a heuristic. Later you can replace it with
        map-specific exit detection or shared-memory exit data.
        """
        if self.goal_position is not None:
            return self.goal_position

        if self.x is not None and self.y is not None:
            self.estimated_goal = (self.x + 1000.0, self.y + 1000.0)
            return self.estimated_goal

        return (0.0, 0.0)

    def set_goal_position(self, x, y):
        """Manually set the goal position if known."""
        self.goal_position = (x, y)

    # ---------------------------------------------------------
    # Weapon / kill estimation
    # ---------------------------------------------------------

    def _estimate_current_weapon(self, ammo):
        """
        Temporary weapon estimate.

        This is conservative. Real weapon detection should eventually come
        from shared memory, controller weapon tracking, or HUD detection.
        """
        if ammo is not None and ammo <= 0:
            return "melee_unknown"

        return "pistol"

    # ---------------------------------------------------------
    # Game state
    # ---------------------------------------------------------

    def get_game_state(self):
        """Return tracked game variables and deltas."""

        shared = read_player_state()

        if shared is not None:
            self.shared_state_available = True

            self.health = shared.get("health", self.health)

            ammo_data = shared.get("ammo", None)
            if ammo_data is not None:
                if isinstance(ammo_data, (list, tuple)):
                    self.ammo = int(sum(ammo_data))
                else:
                    self.ammo = int(ammo_data)

            self.x = shared.get("x", self.x)
            self.y = shared.get("y", self.y)
            self.angle = shared.get("angle", self.angle)
            self.armor = shared.get("armor", self.armor)
            self.kills = shared.get("kills", self.kills)

        health_delta = self.health - self.prev_health
        ammo_delta = self.ammo - self.prev_ammo

        current_weapon = self._estimate_current_weapon(self.ammo)

        weapon_delta = 0
        if self.previous_weapon is not None and current_weapon != self.previous_weapon:
            weapon_delta = 1

        current_kills = self.kills if self.kills is not None else 0
        kill_delta = max(0, current_kills - self.prev_kills)

        state = {
            "health": self.health,
            "ammo": self.ammo,
            "armor": self.armor,

            "health_delta": health_delta,
            "ammo_delta": ammo_delta,

            "weapon": current_weapon,
            "weapon_delta": weapon_delta,

            "kills": current_kills,
            "kill_delta": kill_delta,

            "x": self.x,
            "y": self.y,
            "angle": self.angle,

            "shared_state_available": self.shared_state_available,
        }

        self.prev_health = self.health
        self.prev_ammo = self.ammo
        self.prev_kills = current_kills
        self.previous_weapon = current_weapon

        return state

    # ---------------------------------------------------------
    # Frame observation
    # ---------------------------------------------------------

    # ---------------------------------------------------------
# Frame observation
# ---------------------------------------------------------

    def get_frame(self):
        raw, w, h = frame_cache.get_raw()

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (OUT_W, OUT_H))

        return frame.astype(np.uint8).copy()



    def build(self):
        return self.get_frame()