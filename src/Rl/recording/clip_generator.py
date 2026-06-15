import pickle
import os


class ClipGenerator:

    def __init__(self, clip_length=60):

        self.clip_length = clip_length
        self.buffer = []
        self.clip_index = 0

        os.makedirs("clips", exist_ok=True)


    def record(self, frame, action):

        self.buffer.append((frame, action))

        if len(self.buffer) >= self.clip_length:

            self.save_clip()


    def save_clip(self):

        filename = f"clips/clip_{self.clip_index}.pkl"

        with open(filename, "wb") as f:

            pickle.dump(self.buffer, f)

        print("Saved:", filename)

        self.buffer = []
        self.clip_index += 1