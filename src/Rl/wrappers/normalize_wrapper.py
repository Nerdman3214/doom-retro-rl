import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0
