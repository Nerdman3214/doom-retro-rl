import numpy as np

class FrameProcessor:
    def __init__(self):
        pass  # Removed invalid self assignments

    def channel_diff_mean(self, frame, r_idx, g_idx, b_idx):
        r = frame[:, :, r_idx - 1].astype(np.float32)
        g = frame[:, :, g_idx - 1].astype(np.float32)
        return np.mean(r - g)

    def centre_channel_diff_mean(self, frame, width, height, channel_a, region_size, channel_b, channel_c):
        cx1 = width // 2 - region_size
        cx2 = width // 2 + region_size
        cy1 = height // 2 - region_size
        cy2 = height // 2 + region_size
        region = frame[cy1:cy2, cx1:cx2]
        a = region[:, :, channel_a - 1].astype(np.float32)
        b = region[:, :, channel_b - 1].astype(np.float32)
        return np.mean(a - b)

    def floor_green_ratio(self, frame):
        h, w = frame.shape[:2]
        floor = frame[int(h * 0.65):h, :]
        green = floor[:, :, 1]
        red = floor[:, :, 2]
        mask = (green > red + 20)
        return mask.mean()

    def red_flash_ratio(self, frame):
        red = frame[:, :, 2]
        green = frame[:, :, 1]
        mask = red > green + 40
        return mask.mean()   