import numpy as np
import cv2


class FrameProcessor:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
        self.prev_gray = None

    # -----------------------------
    # CORE PREPROCESS
    # -----------------------------
    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.uint8)

    # -----------------------------
    # MOTION SIGNAL
    # -----------------------------
    def motion(self, frame):
        gray = self.preprocess(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        diff = np.abs(gray.astype(np.int16) - self.prev_gray.astype(np.int16))
        self.prev_gray = gray
        return float(np.mean(diff))

    # -----------------------------
    # CENTER OF MASS (enemy proxy)
    # -----------------------------
    def center_of_mass(self, frame):
        gray = self.preprocess(frame)

        h, w = gray.shape
        y, x = np.indices((h, w))

        total = gray.sum() + 1e-8

        cx = (x * gray).sum() / total
        cy = (y * gray).sum() / total

        return (cx / w) * 2 - 1, (cy / h) * 2 - 1

    # -----------------------------
    # CHANNEL DIFFERENCE FEATURES
    # -----------------------------
    def channel_diff_mean(self, frame, c1, c2, c3):
        # simple RGB/BGR channel contrast signal
        return float(np.mean(frame[:, :, c1].astype(np.float32) - frame[:, :, c2]))

    def centre_channel_diff_mean(self, frame, w, h, c1, crop, c2, c3):
        h0, w0 = frame.shape[:2]
        cx, cy = w0 // 2, h0 // 2

        x1, x2 = cx - crop, cx + crop
        y1, y2 = cy - crop, cy + crop

        crop_img = frame[y1:y2, x1:x2]
        return self.channel_diff_mean(crop_img, c1, c2, c3)

    def centre_channel_count(self, frame, w, h, c1, crop, c2, c3, threshold):
        h0, w0 = frame.shape[:2]
        cx, cy = w0 // 2, h0 // 2

        x1, x2 = cx - crop, cx + crop
        y1, y2 = cy - crop, cy + crop

        crop_img = frame[y1:y2, x1:x2]

        diff = crop_img[:, :, c1].astype(np.int16) - crop_img[:, :, c2].astype(np.int16)
        return float(np.sum(diff > threshold))

    def centre_channel_x_mean(self, frame, w, h, c1, crop, c2, c3, threshold):
        h0, w0 = frame.shape[:2]
        cx, cy = w0 // 2, h0 // 2

        x1, x2 = cx - crop, cx + crop
        y1, y2 = cy - crop, cy + crop

        crop_img = frame[y1:y2, x1:x2]

        mask = (crop_img[:, :, c1].astype(np.int16) - crop_img[:, :, c2]) > threshold

        if not np.any(mask):
            return 0

        xs = np.where(mask)[1]
        return float(np.mean(xs))

    # -----------------------------
    # SIMPLE COLOR SIGNALS (YOU WERE MISSING THESE)
    # -----------------------------
    def floor_green_ratio(self, frame):
        green = frame[:, :, 1]
        return float(np.mean(green > 120))

    def red_flash_ratio(self, frame):
        red = frame[:, :, 2]
        return float(np.mean(red > 150))

    # -----------------------------
    # MAIN EXTRACTOR
    # -----------------------------
    def extract(self, frame):
        obs = self.preprocess(frame)
        motion = self.motion(frame)
        cx, cy = self.center_of_mass(frame)

        return obs, motion, cx, cy