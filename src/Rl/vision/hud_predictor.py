import cv2
import numpy as np


class HudPredictor:
    """
    Lightweight HUD perception.

    This is not OCR yet. It gives the agent structured HUD awareness:
    - health/ammo/armor buckets from game_state when available
    - visual HUD color statistics from the bottom HUD region
    - low health / low ammo / damage flash flags

    This keeps HUD understanding separate from scene/object vision so the
    scene model does not confuse the HUD face, numbers, or weapon sprite
    with enemies/walls.
    """

    def __init__(self, hud_top_ratio=0.78):
        self.hud_top_ratio = hud_top_ratio

    def _ensure_bgr(self, frame):
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return None

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def crop_hud(self, frame):
        frame = self._ensure_bgr(frame)

        if frame is None:
            return None

        h, w = frame.shape[:2]
        y1 = int(h * self.hud_top_ratio)

        return frame[y1:, :]

    def _bucket(self, value, low, medium):
        if value is None:
            return "unknown"

        try:
            value = float(value)
        except Exception:
            return "unknown"

        if value <= low:
            return "low"

        if value <= medium:
            return "medium"

        return "high"

    def _safe_number(self, game_state, key, default=None):
        value = game_state.get(key, default)

        try:
            return float(value)
        except Exception:
            return default

    def predict(self, frame, game_state=None):
        game_state = game_state or {}

        hud = self.crop_hud(frame)

        result = {
            "visible": False,
            "health": None,
            "ammo": None,
            "armor": None,
            "health_bucket": "unknown",
            "ammo_bucket": "unknown",
            "armor_bucket": "unknown",
            "low_health": False,
            "low_ammo": False,
            "armor_present": False,
            "damage_flash": False,
            "visual": {
                "red_ratio": 0.0,
                "green_ratio": 0.0,
                "blue_ratio": 0.0,
                "yellow_ratio": 0.0,
                "bright_ratio": 0.0,
            },
        }

        # Use existing game_state values first. These are reliable and
        # correspond to information a human sees on the HUD.
        health = self._safe_number(game_state, "health")
        ammo = self._safe_number(game_state, "ammo")
        armor = self._safe_number(game_state, "armor", 0)

        result["health"] = health
        result["ammo"] = ammo
        result["armor"] = armor

        result["health_bucket"] = self._bucket(health, low=25, medium=70)
        result["ammo_bucket"] = self._bucket(ammo, low=5, medium=25)
        result["armor_bucket"] = self._bucket(armor, low=0, medium=50)

        result["low_health"] = result["health_bucket"] == "low"
        result["low_ammo"] = result["ammo_bucket"] == "low"
        result["armor_present"] = armor is not None and armor > 0

        if hud is None or hud.size == 0:
            return result

        result["visible"] = True

        b = hud[:, :, 0].astype(np.int16)
        g = hud[:, :, 1].astype(np.int16)
        r = hud[:, :, 2].astype(np.int16)

        red_mask = (r > 120) & (r > g + 25) & (r > b + 25)
        green_mask = (g > 120) & (g > r + 15) & (g > b + 15)
        blue_mask = (b > 120) & (b > r + 15) & (b > g + 15)
        yellow_mask = (r > 130) & (g > 110) & (b < 90)
        bright_mask = (r + g + b) > 420

        red_ratio = float(np.mean(red_mask))
        green_ratio = float(np.mean(green_mask))
        blue_ratio = float(np.mean(blue_mask))
        yellow_ratio = float(np.mean(yellow_mask))
        bright_ratio = float(np.mean(bright_mask))

        result["visual"] = {
            "red_ratio": red_ratio,
            "green_ratio": green_ratio,
            "blue_ratio": blue_ratio,
            "yellow_ratio": yellow_ratio,
            "bright_ratio": bright_ratio,
        }

        # Damage flash can affect the HUD/game view with a red overlay.
        result["damage_flash"] = red_ratio > 0.18

        return result
