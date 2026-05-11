import numpy as np
import cv2


class FrameProcessor:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height

        self.prev_frame = None

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.uint8)
    
    def enemy_heatmap(self, frame):
        gray = self.preprocess(frame)

        # simple proxy (replace later with real detector)
        heatmap = (gray > 120).astype(np.float32)

        return heatmap

    def motion(self, frame):
        gray = self.preprocess(frame)

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0

        diff = np.abs(gray.astype(np.int16) - self.prev_frame.astype(np.int16))
        self.prev_frame = gray

        return float(np.mean(diff))
    
    def attention_crop(self, frame, cx, cy, size=42):
        h, w = frame.shape[:2]

        x = int((cx + 1) * w / 2)
        y = int((cy + 1) * h / 2)

        x1 = max(0, x - size)
        x2 = min(w, x + size)
        y1 = max(0, y - size)
        y2 = min(h, y + size)

        return frame[y1:y2, x1:x2]

    def center_of_mass(self, frame):
        gray = self.preprocess(frame)

        h, w = gray.shape
        y, x = np.indices((h, w))

        total = gray.sum() + 1e-8

        cx = (x * gray).sum() / total
        cy = (y * gray).sum() / total

        return (cx / w) * 2 - 1, (cy / h) * 2 - 1

    def extract(self, frame):
        obs = self.preprocess(frame)
        motion = self.motion(frame)
        cx, cy = self.center_of_mass(frame)

        return obs, motion, cx, cy