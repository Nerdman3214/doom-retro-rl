import cv2
import numpy as np


class WallSensor:
    """
    Lightweight pseudo-depth / obstacle sensor for Doom Retro screenshots.

    This does not see through walls. It estimates nearby obstacles from the
    visible gameplay frame using edges, darkness, saturation, and motion-safe
    screen regions.

    Outputs:
    - left/front/right wall ratios
    - near_wall
    - safest_escape_direction
    - obstacle_pressure
    """

    def __init__(self, hud_cut_ratio=0.80):
        self.hud_cut_ratio = hud_cut_ratio

    def _ensure_bgr(self, frame):
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return None

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def gameplay_crop(self, frame):
        frame = self._ensure_bgr(frame)

        if frame is None:
            return None

        h, w = frame.shape[:2]
        return frame[: int(h * self.hud_cut_ratio), :]

    def _region_pressure(self, region):
        if region is None or region.size == 0:
            return 0.0

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Low texture/dark/edge-heavy regions often correspond to walls,
        # doors, pillars, or close geometry in Doom.
        edges = cv2.Canny(gray, 80, 160)

        dark_ratio = float(np.mean(gray < 45))
        bright_ratio = float(np.mean(gray > 210))
        edge_ratio = float(np.mean(edges > 0))

        # Near walls often fill the view with large same-color surfaces.
        # This simple score is intentionally conservative.
        pressure = (
            0.45 * edge_ratio
            + 0.35 * dark_ratio
            + 0.20 * bright_ratio
        )

        return float(np.clip(pressure, 0.0, 1.0))

    def analyze(self, frame):
        crop = self.gameplay_crop(frame)

        if crop is None:
            return {
                "left_ratio": 0.0,
                "front_ratio": 0.0,
                "right_ratio": 0.0,
                "near_wall": False,
                "front_blocked": False,
                "safest_escape_direction": "unknown",
                "obstacle_pressure": 0.0,
            }

        h, w = crop.shape[:2]

        # Focus on lower-middle gameplay area where walls/half-walls block movement.
        y1 = int(h * 0.35)
        y2 = int(h * 0.95)

        left = crop[y1:y2, 0:int(w * 0.33)]
        front = crop[y1:y2, int(w * 0.33):int(w * 0.67)]
        right = crop[y1:y2, int(w * 0.67):w]

        left_ratio = self._region_pressure(left)
        front_ratio = self._region_pressure(front)
        right_ratio = self._region_pressure(right)

        obstacle_pressure = max(left_ratio, front_ratio, right_ratio)

        front_blocked = front_ratio >= 0.42
        near_wall = obstacle_pressure >= 0.45

        # Escape away from the side with more pressure.
        if left_ratio > right_ratio + 0.05:
            safest = "right"
        elif right_ratio > left_ratio + 0.05:
            safest = "left"
        else:
            safest = "back_or_turn"

        return {
            "left_ratio": left_ratio,
            "front_ratio": front_ratio,
            "right_ratio": right_ratio,
            "near_wall": near_wall,
            "front_blocked": front_blocked,
            "safest_escape_direction": safest,
            "obstacle_pressure": obstacle_pressure,
        }
