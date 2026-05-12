import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import frame_cache
from observation.frame_processor import FrameProcessor


class EnemyDetector:
    def __init__(self):
        self.fp = FrameProcessor()

        self.previous_enemy_pixels = None
        self.previous_enemy_center = None
        self.motion_momentum = 0.0
        self.confidence = 0.0

    def reset(self):
        self.previous_enemy_pixels = None
        self.previous_enemy_center = None
        self.motion_momentum = 0.0
        self.confidence = 0.0

    def capture(self):
        return frame_cache.get_bgr()

    def detect_enemy_presence(self, frame):
        """
        Simple red-over-green enemy proxy.
        Not perfect, but stable enough for reward shaping.
        """
        global_red = self.fp.channel_diff_mean(frame, c1=2, c2=1)
        center_red = self.fp.centre_channel_diff_mean(frame, c1=2, c2=1, radius=40)

        score = global_red * 0.35 + center_red * 0.65
        detected = score > 15.0

        if detected:
            self.confidence = min(10.0, self.confidence + 1.0)
        else:
            self.confidence *= 0.90

        return detected

    def detect_enemy_centered(self, frame):
        center_red = self.fp.centre_channel_diff_mean(frame, c1=2, c2=1, radius=24)
        return center_red > 20.0

    def detect_enemy_size_growth(self, frame=None):
        if frame is None:
            frame = self.capture()

        enemy_pixels = self.fp.centre_channel_count(
            frame,
            c1=2,
            c2=1,
            threshold=20,
            radius=45,
        )

        if self.previous_enemy_pixels is None:
            self.previous_enemy_pixels = enemy_pixels
            return 0.0

        growth = float(enemy_pixels - self.previous_enemy_pixels)
        self.previous_enemy_pixels = enemy_pixels

        smoothed = growth * 0.5 + self.motion_momentum * 0.5
        self.motion_momentum = smoothed

        return smoothed

    def detect_enemy_motion_direction(self, frame=None):
        if frame is None:
            frame = self.capture()

        current_center = self.fp.centre_channel_x_mean(
            frame,
            c1=2,
            c2=1,
            threshold=20,
            radius=45,
        )

        if current_center is None:
            return 0.0

        if self.previous_enemy_center is None:
            self.previous_enemy_center = current_center
            return 0.0

        movement = current_center - self.previous_enemy_center
        self.previous_enemy_center = current_center

        return float(movement)

    def detect_threat_level(self, frame):
        threat = 0.0

        if self.detect_enemy_centered(frame):
            threat += 1.0

        if self.detect_enemy_size_growth(frame) > 5.0:
            threat += 1.0

        return threat

    # Compatibility helpers
    def red_dominance(self):
        self.confidence = min(10.0, self.confidence + 1.0)
        return self.confidence

    def humanoid_ratio(self):
        self.confidence = min(10.0, self.confidence + 1.0)
        return self.confidence