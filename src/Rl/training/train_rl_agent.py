import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.doom_env import DoomEnv
from wrappers.normalize_wrapper import NormalizeObservationWrapper


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "doom_rl_agent")
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(launch_doom=True, record=False):
    env = DoomEnv(
        launch_doom=launch_doom,
        record=record,
    )

    time.sleep(2)

    env = NormalizeObservationWrapper(env)
    return env


env = make_env(launch_doom=True, record=False)

ppo_kwargs = dict(
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=6,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.005,
    vf_coef=0.7,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        normalize_images=False,
    ),
    tensorboard_log=os.path.join(ROOT_DIR, "tensorboard_logs", "doom_ppo"),
)

RESUME = "--resume" in sys.argv

if RESUME and os.path.exists(CHECKPOINT + ".zip"):
    print(f"Resuming from {CHECKPOINT}.zip ...")
    model = PPO.load(CHECKPOINT, env=env)
else:
    if RESUME:
        print("No checkpoint found — starting fresh.")

    model = PPO(
        "CnnPolicy",
        env,
        **ppo_kwargs,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="doom_rl_agent",
    verbose=1,
)

try:
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=False,
    )

except KeyboardInterrupt:
    print("\nTraining interrupted — saving current model...")

finally:
    model.save(CHECKPOINT)
    env.close()
    print(f"Saved to {CHECKPOINT}.zip")