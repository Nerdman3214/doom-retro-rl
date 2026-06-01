import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from env.vizdoom_env import VizDoomEnv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
TENSORBOARD_DIR = os.path.join(ROOT_DIR, "tensorboard_logs", "vizdoom_recurrent_ppo")

CHECKPOINT = os.path.join(CHECKPOINT_DIR, "vizdoom_recurrent_ppo_agent")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


class ProgressPrintCallback(BaseCallback):
    def __init__(self, print_freq=1000):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[train_vizdoom] timesteps={self.num_timesteps}")
        return True


def make_env():
    """
    ViZDoom training environment.

    Compact observation:
        shape = (4, 84, 84)

    Channels:
        0 = red
        1 = green
        2 = blue
        3 = depth

    This is the lower-RAM training mode.
    Labels and automap are reserved for geometry dataset collection.
    """

    env = VizDoomEnv(
        visible=True,
        use_depth=True,
        use_labels=False,
        use_automap=False,
        observation_mode="compact",
        frame_skip=4,
        max_episode_steps=1000,
    )

    env = Monitor(env, filename=os.path.join(LOG_DIR, "vizdoom_monitor.csv"))

    return env


env = make_env()

policy_kwargs = dict(
    lstm_hidden_size=256,
    n_lstm_layers=1,
    shared_lstm=True,
    enable_critic_lstm=False,
    normalize_images=False,
)

model_kwargs = dict(
    learning_rate=2.5e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=4,
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
        print("No ViZDoom recurrent checkpoint found — starting fresh.")

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        **model_kwargs,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="vizdoom_recurrent_ppo_agent",
    verbose=1,
)

try:
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, ProgressPrintCallback(print_freq=1000)],
        progress_bar=True,
    )

except KeyboardInterrupt:
    print("\nTraining interrupted — saving current ViZDoom model...")

finally:
    model.save(CHECKPOINT)
    env.close()
    print(f"Saved to {CHECKPOINT}.zip")
