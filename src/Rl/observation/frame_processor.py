import numpy as np
import cv2


class FrameProcessor:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
        self.prev_gray = None

    # ---------------------------------------------------------
    # Core preprocessing
    # ---------------------------------------------------------

    def preprocess(self, frame):
        """
        Converts raw BGR/RGB frame into 84x84 grayscale uint8.
        Most screenshot/cv2 paths are BGR, so this uses BGR2GRAY.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.uint8)

    def extract(self, frame):
        obs = self.preprocess(frame)
        motion = self.motion(frame)
        cx, cy = self.center_of_mass(frame)
        return obs, motion, cx, cy

    # ---------------------------------------------------------
    # Motion / attention
    # ---------------------------------------------------------

    def motion(self, frame):
        gray = self.preprocess(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        diff = np.abs(gray.astype(np.int16) - self.prev_gray.astype(np.int16))
        self.prev_gray = gray
        return float(np.mean(diff))

    def center_of_mass(self, frame):
        gray = self.preprocess(frame)
        h, w = gray.shape

        y, x = np.indices((h, w))
        total = float(gray.sum()) + 1e-8

        cx = float((x * gray).sum()) / total
        cy = float((y * gray).sum()) / total

        norm_x = (cx / w) * 2.0 - 1.0
        norm_y = (cy / h) * 2.0 - 1.0

        return norm_x, norm_y

    def attention_crop(self, frame, cx, cy, size=42):
        h, w = frame.shape[:2]

        x = int((cx + 1.0) * w / 2.0)
        y = int((cy + 1.0) * h / 2.0)

        x1 = max(0, x - size)
        x2 = min(w, x + size)
        y1 = max(0, y - size)
        y2 = min(h, y + size)

        return frame[y1:y2, x1:x2]

    # ---------------------------------------------------------
    # Enemy/color feature helpers
    # BGR convention:
    #   channel 0 = blue
    #   channel 1 = green
    #   channel 2 = red
    # ---------------------------------------------------------

    def channel_diff_mean(self, frame, c1=2, c2=1):
        if frame is None or frame.size == 0:
            return 0.0

        diff = frame[:, :, c1].astype(np.int16) - frame[:, :, c2].astype(np.int16)
        return float(np.mean(diff))

    def centre_channel_diff_mean(self, frame, c1=2, c2=1, radius=32):
        region = self._center_crop(frame, radius)
        return self.channel_diff_mean(region, c1=c1, c2=c2)

    def centre_channel_count(self, frame, c1=2, c2=1, threshold=20, radius=40):
        region = self._center_crop(frame, radius)

        if region is None or region.size == 0:
            return 0

        diff = region[:, :, c1].astype(np.int16) - region[:, :, c2].astype(np.int16)
        return int(np.sum(diff > threshold))

    def centre_channel_x_mean(self, frame, c1=2, c2=1, threshold=20, radius=40):
        region = self._center_crop(frame, radius)

        if region is None or region.size == 0:
            return None

        diff = region[:, :, c1].astype(np.int16) - region[:, :, c2].astype(np.int16)
        mask = diff > threshold

        if not np.any(mask):
            return None

        xs = np.where(mask)[1]
        return float(np.mean(xs))

    def floor_green_ratio(self, frame, threshold=120):
        if frame is None or frame.size == 0:
            return 0.0

        green = frame[:, :, 1]
        return float(np.mean(green > threshold))

    def red_flash_ratio(self, frame, threshold=150):
        if frame is None or frame.size == 0:
            return 0.0

        red = frame[:, :, 2]
        return float(np.mean(red > threshold))

    def enemy_heatmap(self, frame):
        if frame is None or frame.size == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        resized = cv2.resize(frame, (self.width, self.height))
        red = resized[:, :, 2].astype(np.int16)
        green = resized[:, :, 1].astype(np.int16)

        heatmap = (red - green) > 20
        return heatmap.astype(np.float32)

    def _center_crop(self, frame, radius):
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)

        return frame[y1:y2, x1:x2]