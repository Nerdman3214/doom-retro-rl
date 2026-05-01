import os
import numpy as np


class TrajectoryLogger:

    BATCH_SIZE = 128

    def __init__(self):

        self.trajectory = []

        os.makedirs("trajectories", exist_ok=True)

        self.counter = len(os.listdir("trajectories"))


    def record(self, frame, action, reward):

        self.trajectory.append((frame, action, reward))

        if len(self.trajectory) >= self.BATCH_SIZE:
            self.save()


    def save(self):

        if len(self.trajectory) == 0:

            return


        filename = f"trajectories/traj_{self.counter}.npz"


        frames = []

        actions = []

        rewards = []


        for f, a, r in self.trajectory:

            frames.append(f)

            actions.append(a)

            rewards.append(r)


        np.savez(

            filename,

            frames=np.array(frames),

            actions=np.array(actions),

            rewards=np.array(rewards)

        )


        self.counter += 1

        self.trajectory = []

        self.trajectory = []