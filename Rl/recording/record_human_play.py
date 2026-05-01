import pickle

class HumanRecorder:

    def __init__(self):

        self.trajectory = []

    def record(self, frame, action):

        self.trajectory.append(
            (frame, action)
        )

    def save(self, filename="human_traj.pkl"):

        with open(filename, "wb") as f:

            pickle.dump(self.trajectory, f)

        print("Saved human trajectory.")