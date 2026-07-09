import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.doom_env import DoomEnv
from wrappers.normalize_wrapper import NormalizeObservationWrapper


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "doom_recurrent_ppo")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(launch_doom=True):
    env = DoomEnv(launch_doom=launch_doom)
    time.sleep(2)
    env = NormalizeObservationWrapper(env)
    env = Monitor(env, filename=os.path.join(LOG_DIR, "recurrent_train_monitor.csv"))
    return env


env = make_env(launch_doom=True)

ppo_kwargs = dict(
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=5,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.005,
    vf_coef=0.7,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        normalize_images=False,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    ),
    tensorboard_log=os.path.join(ROOT_DIR, "tensorboard_logs", "doom_recurrent_ppo"),
)

RESUME = "--resume" in sys.argv

if RESUME and os.path.exists(CHECKPOINT + ".zip"):
    print(f"Resuming from {CHECKPOINT}.zip ...")
    model = RecurrentPPO.load(CHECKPOINT, env=env)
else:
    if RESUME:
        print("No checkpoint found — starting fresh.")

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        **ppo_kwargs,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="doom_recurrent_ppo",
    verbose=1,
)

try:
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

except KeyboardInterrupt:
    print("\nTraining interrupted — saving current recurrent model...")

finally:
    model.save(CHECKPOINT)
    env.close()
    print(f"Saved to {CHECKPOINT}.zip")