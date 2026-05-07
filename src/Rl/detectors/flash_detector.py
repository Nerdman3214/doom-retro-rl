import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import frame_cache
import frame_processor


class FlashDetector:

    def __init__(self):
        self.previous_frame = None

    def capture(self):
        return frame_cache.get_bgr()

    def detect_player_damage(self, frame):
        # mean of R channel (index 2 in BGR)
        return float(np.mean(frame[:, :, 2])) > 150

    def detect_enemy_damage(self, frame):
        brightness = frame_processor.mean_brightness(frame)
        return brightness > 180

    def detect_pickup_flash(self, frame):
        brightness = frame_processor.mean_brightness(frame)
        return 130 < brightness < 160
