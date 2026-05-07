import pickle
import os
from collections import deque


class SmartClipGenerator:

    def __init__(self, pre_frames=30, post_frames=60):

        self.pre_frames = pre_frames
        self.post_frames = post_frames

        self.frame_buffer = deque(maxlen=pre_frames)

        self.recording = False
        self.post_counter = 0

        self.clip = []

        self.clip_index = 0

        os.makedirs("clips", exist_ok=True)


    def observe(self, frame, action):

        self.frame_buffer.append((frame, action))

        if self.recording:

            self.clip.append((frame, action))

            self.post_counter -= 1

            if self.post_counter <= 0:

                self.save_clip()

                self.recording = False


    def trigger_event(self):

        if not self.recording:

            self.recording = True

            self.clip = list(self.frame_buffer)

            self.post_counter = self.post_frames


    def save_clip(self):

        filename = f"clips/event_clip_{self.clip_index}.pkl"

        with open(filename, "wb") as f:

            pickle.dump(self.clip, f)

        print("Saved smart clip:", filename)

        self.clip_index += 1