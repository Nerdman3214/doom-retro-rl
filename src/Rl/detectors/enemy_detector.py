import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import frame_cache

from observation.frame_processor import FrameProcessor


class EnemyDetector:

    def __init__(self):

        self.fp = FrameProcessor()

        self.previous_enemy_pixels = None
        self.previous_enemy_center = None

        self.motion_momentum = 0.0
        self.confidence = 0.0

    def channel_diff_mean(self, frame, c1=2, c2=1):
        diff = frame[:, :, c1].astype(np.int16) - frame[:, :, c2].astype(np.int16)
        return float(np.mean(diff))

    def capture(self):
        return frame_cache.get_bgr()

    # =========================================================
    # MAIN ENEMY DETECTION
    # =========================================================

    def detect_enemy_presence(self, frame):

        red_dom = self.fp.channel_diff_mean(
            frame,
            c1=2,   # red
            c2=1    # green
        )

        center_dom = self.fp.centre_channel_diff_mean(
            frame,
            c1=2,
            c2=1,
            radius=32
        )

        confidence_score = (
            red_dom * 0.4 +
            center_dom * 0.6
        )

        detected = confidence_score > 18

        if detected:
            self.confidence = min(10.0, self.confidence + 1.0)
        else:
            self.confidence *= 0.90

        return detected

    # =========================================================
    # CROSSHAIR TARGETING
    # =========================================================

    def detect_enemy_centered(self, frame):

        center_strength = self.fp.centre_channel_diff_mean(
            frame,
            c1=2,
            c2=1,
            radius=20
        )

        return center_strength > 22

    # =========================================================
    # ENEMY SIZE CHANGE
    # =========================================================

    def detect_enemy_size_growth(self, frame):

        enemy_pixels = self.fp.centre_channel_count(
            frame,
            c1=2,
            c2=1,
            threshold=20,
            radius=40
        )

        if self.previous_enemy_pixels is None:
            self.previous_enemy_pixels = enemy_pixels
            return 0.0

        growth = enemy_pixels - self.previous_enemy_pixels

        self.previous_enemy_pixels = enemy_pixels

        smoothed = (
            growth * 0.5 +
            self.motion_momentum * 0.5
        )

        self.motion_momentum = smoothed

        return smoothed

    # =========================================================
    # ENEMY MOVEMENT TRACKING
    # =========================================================

    def detect_enemy_motion_direction(self, frame):

        center_x = self.fp.centre_channel_x_mean(
            frame,
            c1=2,
            c2=1,
            threshold=20,
            radius=40
        )

        if center_x is None:
            return 0.0

        if self.previous_enemy_center is None:
            self.previous_enemy_center = center_x
            return 0.0

        movement = center_x - self.previous_enemy_center

        self.previous_enemy_center = center_x

        return movement

    # =========================================================
    # DODGE SIGNAL
    # =========================================================

    def detect_threat_level(self, frame):

        size_growth = self.detect_enemy_size_growth(frame)

        centered = self.detect_enemy_centered(frame)

        threat = 0.0

        if size_growth > 5:
            threat += 1.0

        if centered:
            threat += 1.0

        return threat

    # =========================================================
    # RESET BETWEEN EPISODES
    # =========================================================

    def reset(self):

        self.previous_enemy_pixels = None
        self.previous_enemy_center = None

        self.motion_momentum = 0.0
        self.confidence = 0.0