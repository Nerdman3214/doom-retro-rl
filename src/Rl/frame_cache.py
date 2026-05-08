"""
frame_cache.py

Single screen capture per environment step, shared across all detectors
and the observation builder.  Without this, each detector called mss.grab()
independently — 10+ full-resolution BGRA captures per step (~8 MB each).

Usage
-----
At the start of each env step:
    import frame_cache
    frame_cache.invalidate()

Then anywhere that previously called mss.grab() / np.array(screenshot):
    bgr = frame_cache.get_bgr()     # numpy HxWx3 uint8, BGR
    raw, w, h = frame_cache.get_raw()  # raw BGRA bytes + dimensions
"""

import mss
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils.window_focus import get_doom_window_rect

_sct     = None   # single mss instance
_bgr     = None   # numpy HxWx3 BGR, cached per step
_raw     = None   # raw BGRA bytes, cached per step
_width   = 0
_height  = 0
_monitor = None   # mss monitor dict for the DOOM window


def _ensure_sct():
    global _sct
    if _sct is None:
        _sct = mss.mss()


def _get_monitor():
    """Build an mss monitor dict cropped to the DOOM window.

    Falls back to the full monitor if the window cannot be found.
    Cached after the first successful lookup so we don't call xdotool
    every step.
    """
    global _monitor
    if _monitor is not None:
        return _monitor

    rect = get_doom_window_rect()
    if rect:
        left, top, width, height = rect
        _monitor = {"left": left, "top": top, "width": width, "height": height}
        print(f"[frame_cache] Capturing DOOM window at "
              f"{left},{top} {width}×{height}")
    else:
        _ensure_sct()
        _monitor = _sct.monitors[1]
        print("[frame_cache] DOOM window not found — capturing full monitor.")
    return _monitor


def invalidate():
    """Call at the start of every env step to force a fresh capture."""
    global _bgr, _raw
    _bgr = None
    _raw = None


def reset_monitor():
    """Call if DOOM is restarted so the window geometry is re-detected."""
    global _monitor
    _monitor = None


def _capture():
    """Do the actual screen grab and populate both caches."""
    global _bgr, _raw, _width, _height
    _ensure_sct()
    monitor    = _get_monitor()
    screenshot = _sct.grab(monitor)
    _raw       = bytes(screenshot.raw)      # BGRA bytes from mss ScreenShot
    _width     = screenshot.width
    _height    = screenshot.height
    arr        = np.frombuffer(_raw, dtype=np.uint8).reshape(_height, _width, 4)
    _bgr       = np.ascontiguousarray(arr[:, :, :3])  # drop alpha → BGR


def get_bgr() -> np.ndarray:
    """Return cached full-resolution BGR numpy array (HxWx3 uint8)."""
    if _bgr is None:
        _capture()
    return _bgr


def get_raw():
    """Return (bgra_bytes, width, height).  Bytes are the raw mss buffer."""
    if _raw is None:
        _capture()
    return _raw, _width, _height
