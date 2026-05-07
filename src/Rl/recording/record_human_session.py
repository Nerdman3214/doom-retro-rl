import sys
import os
import time
import pickle

import numpy as np
import mss
import cv2
from pynput import keyboard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.window_focus import get_doom_window_rect

OUTPUT_FILE = "human_demo.pkl"
RECORD_FPS = 5                    # lower FPS = far less data per second
DOWNSCALE_SIZE = (160, 120)       # width x height
SCREEN_CHANGE_THRESHOLD = 10      # skip frame if pixel diff below this


class HumanSessionRecorder:
    """
    Records screen + keypresses to disk in a streaming pickle file.

    Each sample is written immediately to disk as a (frame, actions) tuple
    so RAM usage stays near-zero regardless of session length.
    The file format is a stream of pickled objects readable by iterating
    until EOFError — fully compatible with convert_human_demo_to_npy.py.
    """

    def __init__(self):
        self.current_keys = set()
        self.running = True
        self.sct = mss.mss()

        # Capture only the DOOM window, not the whole desktop.
        rect = get_doom_window_rect()
        if rect:
            left, top, width, height = rect
            self.monitor = {"left": left, "top": top,
                            "width": width, "height": height}
            print(f"Recording DOOM window at {left},{top} {width}×{height}")
        else:
            self.monitor = self.sct.monitors[1]
            print("WARNING: DOOM window not found — recording full monitor.")
            print("Make sure DOOM Retro is running before starting a recording.")

        self._prev_frame_bytes = None
        self._frame_count = 0
        self._out_file = open(OUTPUT_FILE, "wb")
        self._pickler = pickle.Pickler(self._out_file)

    def on_press(self, key):
        self.current_keys.add(str(key))

    def on_release(self, key):
        self.current_keys.discard(str(key))

    def _grab_frame(self):
        raw = self.sct.grab(self.monitor)
        frame = np.array(raw)[:, :, :3]                     # drop alpha
        frame = cv2.resize(frame, DOWNSCALE_SIZE)            # downscale
        return frame

    def _should_save(self, frame):
        """Return True if the frame is significantly different from previous."""
        fb = frame.tobytes()
        if self._prev_frame_bytes is None or self.current_keys:
            self._prev_frame_bytes = fb
            return True
        # Fast pixel-diff check using numpy abs mean
        prev = np.frombuffer(self._prev_frame_bytes, dtype=np.uint8)
        curr = np.frombuffer(fb, dtype=np.uint8)
        diff = float(np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16))))
        self._prev_frame_bytes = fb
        return diff > SCREEN_CHANGE_THRESHOLD

    def record_screen(self):
        while self.running:
            frame = self._grab_frame()
            if self._should_save(frame):
                self._pickler.dump((frame, list(self.current_keys)))
                self._frame_count += 1
                if self._frame_count % 50 == 0:
                    self._out_file.flush()    # periodic flush to disk
            time.sleep(1.0 / RECORD_FPS)

    def start(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        listener.start()
        print(
            f"Recording started at {RECORD_FPS} FPS, "
            f"size {DOWNSCALE_SIZE}. Play DOOM Retro. "
            "Press Ctrl+C here to stop and save."
        )
        try:
            self.record_screen()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self._close()

    def _close(self):
        self._out_file.flush()
        self._out_file.close()
        print(f"Saved {self._frame_count} frames to {OUTPUT_FILE}")


if __name__ == "__main__":
    recorder = HumanSessionRecorder()
    recorder.start()

