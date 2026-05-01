import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import frame_cache
import frame_processor


class EnemyDetector:

    def __init__(self):
        self.previous_enemy_pixels = None
        self.momentum = 0
        self.previous_enemy_center = None

    def capture(self):
        return frame_cache.get_bgr()

    def detect_enemy_presence(self, frame):
        # BGR: ch2=R, ch1=G → mean(R - G) > 15
        dom = frame_processor.channel_diff_mean(frame, 3, 2, 1)
        return dom > 15

    def detect_enemy_centered(self, frame):
        h, w = frame.shape[:2]
        dom = frame_processor.centre_channel_diff_mean(frame, w, h, 3, 40, 2, 1)
        return dom > 20

    def detect_enemy_size_growth(self):
        frame = self.capture()
        h, w, _ = frame.shape
        enemy_pixels = frame_processor.centre_channel_count(frame, w, h, 3, 40, 2, 1, 20)

        if self.previous_enemy_pixels is None:
            self.previous_enemy_pixels = enemy_pixels
            return 0

        growth = enemy_pixels - self.previous_enemy_pixels
        self.previous_enemy_pixels = enemy_pixels
        smoothed = growth * 0.5 + self.momentum * 0.5
        self.momentum = growth
        return smoothed

    def detect_enemy_motion_direction(self):
        frame = self.capture()
        h, w, _ = frame.shape
        current_center = frame_processor.centre_channel_x_mean(
            frame, w, h, 3, 40, 2, 1, 20
        )

        if current_center is None:
            return 0

        if self.previous_enemy_center is None:
            self.previous_enemy_center = current_center
            return 0

        movement = current_center - self.previous_enemy_center
        self.previous_enemy_center = current_center
        return movement