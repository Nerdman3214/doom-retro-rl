import collections

import numpy as np


class FrameStack:

    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = collections.deque(maxlen=stack_size)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame.astype(np.uint8))
        return self._get_stacked()

    def add_frame(self, frame):
        self.frames.append(frame.astype(np.uint8))
        return self._get_stacked()

    def _get_stacked(self):
        stacked = np.concatenate(list(self.frames), axis=2)
        return np.transpose(stacked, (2, 0, 1))
