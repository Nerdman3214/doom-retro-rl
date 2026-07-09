import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import frame_cache
import frame_processor


class DoorDetector:

    def __init__(self):
        self.previous_frame_bytes = None

    def capture(self):
        return frame_cache.get_bgr()

    def detect_door_motion(self):
        frame = self.capture()
        frame_bytes = bytes(frame)

        if self.previous_frame_bytes is None:
            self.previous_frame_bytes = frame_bytes
            return False

        diff = frame_processor.mean_abs_diff(frame_bytes, self.previous_frame_bytes)
        self.previous_frame_bytes = frame_bytes
        return diff > 12
