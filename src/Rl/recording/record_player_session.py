import pickle
from pynput import keyboard


class PlayerRecorder:

    def __init__(self):

        self.trajectory = []
        self.current_keys = set()


    def on_press(self, key):

        self.current_keys.add(str(key))


    def on_release(self, key):

        if str(key) in self.current_keys:

            self.current_keys.remove(str(key))


    def record_frame(self, frame):

        action_snapshot = list(self.current_keys)

        self.trajectory.append(
            (frame, action_snapshot)
        )


    def record(self, frame, action):
        # For compatibility with old code, just call record_frame
        self.record_frame(frame)


    def save(self):

        with open("human_demo.pkl", "wb") as f:

            pickle.dump(self.trajectory, f)

        print("Saved human gameplay dataset.")