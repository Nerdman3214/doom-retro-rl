
from pathlib import Path
import ctypes

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "native" / "libvizdoom_frame_packer.so"


class NativeFramePacker:
    """
    Small ctypes wrapper around native/libvizdoom_frame_packer.so.

    Purpose:
    - reduce Python-side image processing overhead
    - compact ViZDoom RGB + depth into one small uint8 tensor
    - keep geometry/debug buffers separate from RL training buffers
    """

    def __init__(self, out_size=(84, 84)):
        self.out_w, self.out_h = out_size
        self.available = LIB_PATH.exists()
        self.lib = None

        if not self.available:
            print(f"[native_packer] Missing {LIB_PATH}, using Python fallback")
            return

        try:
            self.lib = ctypes.CDLL(str(LIB_PATH))

            self.lib.pack_rgb_hwc_to_chw_u8.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
            ]

            self.lib.pack_gray_u8.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
            ]

            self.lib.pack_depth_f32_to_u8.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
            ]

            self.lib.pack_rgb_depth_compact.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
            ]

            print(f"[native_packer] Loaded {LIB_PATH}")

        except Exception as e:
            print(f"[native_packer] Failed to load native packer: {e}")
            self.available = False
            self.lib = None

    def _to_hwc_rgb(self, rgb):
        arr = np.asarray(rgb)

        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return np.ascontiguousarray(arr.astype(np.uint8))

    def _python_resize_chw(self, img, channels):
        import cv2

        w, h = self.out_w, self.out_h

        if img is None:
            return np.zeros((channels, h, w), dtype=np.uint8)

        arr = np.asarray(img)

        if arr.ndim == 2:
            resized = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
            return resized[None, :, :].astype(np.uint8)

        arr = self._to_hwc_rgb(arr)
        resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        return np.transpose(resized, (2, 0, 1)).astype(np.uint8)

    def pack_rgb(self, rgb):
        rgb = self._to_hwc_rgb(rgb)
        h, w = rgb.shape[:2]

        out = np.zeros((3, self.out_h, self.out_w), dtype=np.uint8)

        if not self.available:
            return self._python_resize_chw(rgb, channels=3)

        self.lib.pack_rgb_hwc_to_chw_u8(
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            h,
            w,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.out_h,
            self.out_w,
        )

        return out

    def pack_depth(self, depth):
        if depth is None:
            return np.zeros((1, self.out_h, self.out_w), dtype=np.uint8)

        depth = np.ascontiguousarray(np.asarray(depth).astype(np.float32))
        h, w = depth.shape[:2]

        out = np.zeros((1, self.out_h, self.out_w), dtype=np.uint8)

        if not self.available:
            import cv2

            lo = float(np.min(depth))
            hi = float(np.max(depth))

            if hi <= lo:
                norm = np.zeros_like(depth, dtype=np.uint8)
            else:
                norm = ((depth - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

            resized = cv2.resize(norm, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            return resized[None, :, :].astype(np.uint8)

        self.lib.pack_depth_f32_to_u8(
            depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            h,
            w,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.out_h,
            self.out_w,
        )

        return out

    def pack_compact_rgb_depth(self, rgb, depth):
        rgb = self._to_hwc_rgb(rgb)
        depth = np.ascontiguousarray(np.asarray(depth).astype(np.float32))

        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]

        out = np.zeros((4, self.out_h, self.out_w), dtype=np.uint8)

        if not self.available:
            out[:3] = self.pack_rgb(rgb)
            out[3:] = self.pack_depth(depth)
            return out

        self.lib.pack_rgb_depth_compact(
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rgb_h,
            rgb_w,
            depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            depth_h,
            depth_w,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.out_h,
            self.out_w,
        )

        return out