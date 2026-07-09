import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from env.doom_env import DoomEnv

env = DoomEnv()

obs, _ = env.reset()

print(obs.shape)

obs, reward, done, truncated, info = env.step(0)

print("step working")