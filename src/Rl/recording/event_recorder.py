import pickle
from pynput import keyboard


class EventRecorder:

    def __init__(self):

        self.trajectory = []
        self.current_keys = set()

        self.current_event = "none"


    def on_press(self, key):

        self.current_keys.add(str(key))


    def on_release(self, key):

        if str(key) in self.current_keys:

            self.current_keys.remove(str(key))


    def set_event(self, event_name):

        self.current_event = event_name


    def record_frame(self, frame):

        action_snapshot = list(self.current_keys)

        self.trajectory.append(
            (frame, action_snapshot, self.current_event)
        )

        self.current_event = "none"


    def save(self):

        with open("human_event_demo.pkl", "wb") as f:

            pickle.dump(self.trajectory, f)

        print("Saved event-tagged dataset.")