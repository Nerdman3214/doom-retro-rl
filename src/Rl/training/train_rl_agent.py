import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.doom_env import DoomEnv
from wrappers.normalize_wrapper import NormalizeObservationWrapper
import time

env = DoomEnv(launch_doom=True)
env.controller.focus_game()
time.sleep(2)
env = NormalizeObservationWrapper(env)

ppo_kwargs = dict(
    verbose=1,
    learning_rate=0.0003,
    # FIX #6: Raised n_steps from 512 → 2048.
    # 512 gave PPO too little rollout data per update for a slow visual
    # environment. 2048 is the standard starting point for CNN policies and
    # produces much more stable gradient estimates.
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    ent_coef=0.01,
    clip_range=0.2,
    policy_kwargs=dict(normalize_images=False),
)

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "doom_rl_agent")
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT)

RESUME = "--resume" in sys.argv

if RESUME and os.path.exists(CHECKPOINT + ".zip"):
    print(f"Resuming from {CHECKPOINT}.zip ...")
    model = PPO.load(CHECKPOINT, env=env)
else:
    if RESUME:
        print("No checkpoint found — starting fresh.")
    model = PPO("CnnPolicy", env, **ppo_kwargs)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="doom_rl_agent",
    verbose=1,
)

try:
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("\nTraining interrupted — saving current model...")
finally:
    model.save(CHECKPOINT)
    print(f"Saved to {CHECKPOINT}.zip")