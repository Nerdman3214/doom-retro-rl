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
LOG_DIR = os.path.join(ROOT_DIR, "logs")
TENSORBOARD_DIR = os.path.join(ROOT_DIR, "tensorboard_logs", "doom_recurrent_ppo")

# New checkpoint name because old PPO checkpoints are not compatible with RecurrentPPO.
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "doom_recurrent_ppo_agent")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


def make_env(launch_doom=True, record=True):
    """
    Training env.

    launch_doom=True means:
    - Automatically launch Doom when the environment is created.
    - Let the controller attach to the existing Doom window.
    - Avoid spawning duplicate hidden Doom processes.

    record=True means:
    - Record the training episodes.
    """

    env = DoomEnv(
        launch_doom=launch_doom,
        record=record,
    )

    time.sleep(2)

    env = NormalizeObservationWrapper(env)
    env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

    return env


env = make_env(launch_doom=True, record=True)

policy_kwargs = dict(
    lstm_hidden_size=512,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
    normalize_images=False,
)

model_kwargs = dict(
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=6,
    gamma=0.997,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    tensorboard_log=TENSORBOARD_DIR,
    verbose=1,
    device="auto",
)

RESUME = "--resume" in sys.argv

if RESUME and os.path.exists(CHECKPOINT + ".zip"):
    print(f"Resuming from {CHECKPOINT}.zip ...")
    model = RecurrentPPO.load(CHECKPOINT, env=env, device="auto")
else:
    if RESUME:
        print("No recurrent checkpoint found — starting fresh.")

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        **model_kwargs,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="doom_recurrent_ppo_agent",
    verbose=1,
)

try:
    model.learn(
        total_timesteps=200_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

except KeyboardInterrupt:
    print("\nTraining interrupted — saving current model...")

finally:
    model.save(CHECKPOINT)
    env.close()
    print(f"Saved to {CHECKPOINT}.zip")