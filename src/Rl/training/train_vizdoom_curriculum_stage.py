import argparse
import os
import sys
from pathlib import Path

# Allow running this file directly:
#   python training/train_vizdoom_curriculum_stage.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

from env.vizdoom_env import VizDoomEnv
from curriculum.task_config import get_task_config, list_task_names


CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list_task_names())
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    task = get_task_config(args.task)

    os.environ["VIZDOOM_TASK"] = task.name

    env = VizDoomEnv(visible=args.visible)
    env = Monitor(env)

    save_path = CHECKPOINT_DIR / f"vizdoom_{task.name}.zip"

    if args.fresh:
        load_path = None
    elif args.load_from:
        load_path = Path(args.load_from)
    else:
        load_path = None

    if load_path and load_path.exists():
        print(f"[curriculum_train] loading previous model: {load_path}")
        model = RecurrentPPO.load(str(load_path), env=env, device="auto")
    else:
        print("[curriculum_train] starting fresh model")
        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            clip_range=0.15,
            tensorboard_log=str(ROOT / "tensorboard_logs" / "vizdoom_curriculum"),
            device="auto",
        )

    print(f"[curriculum_train] task={task.name}")
    print(f"[curriculum_train] desc={task.description}")
    print(f"[curriculum_train] allowed_actions={task.allowed_actions}")
    print(f"[curriculum_train] saving to {save_path}")

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    finally:
        model.save(str(save_path))
        print(f"[curriculum_train] saved {save_path}")
        env.close()


if __name__ == "__main__":
    main()
