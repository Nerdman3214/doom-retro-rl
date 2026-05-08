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
import mss.exception
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from utils.window_focus import get_doom_window_rect

_sct     = None   # single mss instance
_bgr     = None   # numpy HxWx3 BGR, cached per step
_raw     = None   # raw BGRA bytes, cached per step
_width   = 0
_height  = 0
_monitor = None   # mss monitor dict for the DOOM window

# How many times to retry a failed grab before giving up
_MAX_CAPTURE_RETRIES = 5
# Seconds to wait between retries
_RETRY_DELAY = 0.5


def _ensure_sct():
    global _sct
    if _sct is None:
        _sct = mss.mss()


def _get_monitor(force_refresh: bool = False):
    """Build an mss monitor dict cropped to the DOOM window.

    Falls back to the full monitor if the window cannot be found.

    Args:
        force_refresh: If True, ignore the cached value and re-detect the
                       window position. Use this after a reset() so stale
                       coordinates from a previous episode never cause
                       XGetImage() to grab an invalid region.
    """
    global _monitor
    if _monitor is not None and not force_refresh:
        return _monitor

    rect = get_doom_window_rect()
    if rect:
        left, top, width, height = rect

        # Sanity-check: mss will crash with XGetImage() if width/height are 0
        if width <= 0 or height <= 0:
            print(f"[frame_cache] Window rect invalid ({width}x{height}) — "
                  "falling back to full monitor.")
            _ensure_sct()
            _monitor = _sct.monitors[1]
        else:
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
    """Call when DOOM resets/respawns so window geometry is re-detected.

    This is the key fix: the old code cached _monitor forever, so if the
    window moved or wasn't ready yet, every subsequent grab used bad coords.
    Calling this forces _get_monitor() to re-run get_doom_window_rect().
    """
    global _monitor
    _monitor = None


def _capture():
    """Do the actual screen grab and populate both caches.

    Retries up to _MAX_CAPTURE_RETRIES times on XGetImage failure, waiting
    _RETRY_DELAY seconds between attempts and re-detecting the window each
    time.  This handles the race condition where reset() fires before the
    DOOM window is fully composited on screen.
    """
    global _bgr, _raw, _width, _height

    _ensure_sct()

    last_error = None
    for attempt in range(1, _MAX_CAPTURE_RETRIES + 1):
        # On the first attempt use cached coords; on retries force re-detect
        # so we pick up the window if it just finished rendering.
        monitor = _get_monitor(force_refresh=(attempt > 1))
        try:
            screenshot = _sct.grab(monitor)
            _raw    = bytes(screenshot.raw)
            _width  = screenshot.width
            _height = screenshot.height
            arr     = np.frombuffer(_raw, dtype=np.uint8).reshape(
                          _height, _width, 4)
            _bgr    = np.ascontiguousarray(arr[:, :, :3])  # drop alpha → BGR
            return  # success
        except mss.exception.ScreenShotError as e:
            last_error = e
            print(f"[frame_cache] XGetImage() failed "
                  f"(attempt {attempt}/{_MAX_CAPTURE_RETRIES}), "
                  f"retrying in {_RETRY_DELAY}s …")
            # Force monitor re-detection on next loop iteration
            reset_monitor()
            time.sleep(_RETRY_DELAY)

    # All retries exhausted — fall back to full monitor as a last resort
    print("[frame_cache] All retries failed, falling back to full monitor. "
          f"Last error: {last_error}")
    _ensure_sct()
    monitor = _sct.monitors[1]
    screenshot = _sct.grab(monitor)
    _raw    = bytes(screenshot.raw)
    _width  = screenshot.width
    _height = screenshot.height
    arr     = np.frombuffer(_raw, dtype=np.uint8).reshape(_height, _width, 4)
    _bgr    = np.ascontiguousarray(arr[:, :, :3])


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