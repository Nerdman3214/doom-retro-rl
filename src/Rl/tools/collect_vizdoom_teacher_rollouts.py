import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None


CHECKPOINT_PATH = ROOT / "checkpoints" / "vizdoom_recurrent_ppo_agent.zip"
OUTPUT_ROOT = ROOT / "vision_dataset_v2" / "teacher_rollouts_balanced"


def safe_json_value(value):
    """
    Convert numpy/Python values into JSON-safe values.
    """

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]

    return value


def get_action_name(env, action_index):
    """
    Resolve action index into a readable shared action name.

    This prevents metadata from saving action names as "4", "3", etc.
    """

    shared_names = [
        "move_forward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "move_backward",
        "shoot",
        "use",
    ]

    action_index = int(action_index)

    if action_index < len(shared_names):
        return shared_names[action_index]

    return f"action_{action_index}"


def save_frame(frame_path, obs):
    """
    Save a visual frame for later student training/debugging.

    Compact ViZDoom observations are usually CHW with RGB/depth-like channels.
    This saves the first 3 channels when possible.
    """

    arr = obs

    if isinstance(arr, dict):
        if "rgb" in arr:
            arr = arr["rgb"]
        elif "screen" in arr:
            arr = arr["screen"]
        elif "observation" in arr:
            arr = arr["observation"]
        else:
            return False

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] in [3, 4]:
        # CHW -> HWC
        img = arr[:3].transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[2] in [3, 4]:
        img = arr[:, :, :3]
    elif arr.ndim == 2:
        img = arr
    else:
        return False

    img = np.asarray(img)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(frame_path), img)
    return True


def collect_teacher_rollouts(
    episodes=10,
    max_steps_per_episode=1000,
    visible=False,
    use_model=True,
):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    env = VizDoomEnv(
        visible=visible,
        use_depth=True,
        use_labels=True,
        use_automap=False,
        observation_mode="compact",
        frame_skip=4,
        max_episode_steps=max_steps_per_episode,
    )

    model = None
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    if use_model and RecurrentPPO is not None and CHECKPOINT_PATH.exists():
        print(f"[teacher_rollouts] Loading teacher model: {CHECKPOINT_PATH}")
        model = RecurrentPPO.load(str(CHECKPOINT_PATH), env=env, device="auto")
    else:
        print("[teacher_rollouts] No teacher model loaded. Using random actions.")

    for episode in range(episodes):
        episode_dir = OUTPUT_ROOT / f"episode_{episode:06d}"
        frame_dir = episode_dir / "frames"
        episode_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = episode_dir / "metadata.jsonl"

        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        total_reward = 0.0

        print(f"[teacher_rollouts] Episode {episode} started.")

        with metadata_path.open("w", encoding="utf-8") as f:
            for step in range(max_steps_per_episode):
                if model is not None:
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=False,
                    )
                    model_action_index = int(action)
                else:
                    model_action_index = int(env.action_space.sample())

                # --------------------------------------------------
                # Balanced teacher collection
                # --------------------------------------------------
                # The first teacher dataset collapsed to action 4 only.
                # This forces coverage across the first five movement actions
                # while still allowing the teacher model to contribute sometimes.
                cycle = step % 6

                if cycle == 0:
                    action_index = 0
                elif cycle == 1:
                    action_index = 1
                elif cycle == 2:
                    action_index = 2
                elif cycle == 3:
                    action_index = 3
                elif cycle == 4:
                    action_index = 4
                else:
                    action_index = model_action_index

                next_obs, reward, terminated, truncated, info = env.step(action_index)

                total_reward += float(reward)

                action_name = get_action_name(env, action_index)
                reward_debug = info.get("reward_debug", {})
                game_vars = info.get("game_vars", {})

                frame_name = f"step_{step:06d}.png"
                frame_path = frame_dir / frame_name
                frame_saved = save_frame(frame_path, obs)

                row = {
                    "episode": episode,
                    "step": step,
                    "action_index": action_index,
                    "action_name": action_name,
                    "reward": float(reward),
                    "total_reward_so_far": float(total_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "frame": str(frame_path.relative_to(episode_dir)) if frame_saved else None,
                    "backend": info.get("backend", "vizdoom"),
                    "game_vars": game_vars,
                    "reward_debug": reward_debug,
                    "shared_reward": reward_debug.get("total_reward"),
                    "scene_info": reward_debug.get("scene_info"),
                    "object_summary": reward_debug.get("object_summary"),
                    "wall_reward": reward_debug.get("wall_reward"),
                    "movement_reward": reward_debug.get("movement_reward"),
                    "exploration_reward": reward_debug.get("exploration_reward"),
                    "route_reward": reward_debug.get("route_reward"),
                    "combat_reward": reward_debug.get("combat_reward"),
                    "object_scene_reward": reward_debug.get("object_scene_reward"),
                    "curriculum_stage": reward_debug.get("curriculum_stage"),
                }

                f.write(json.dumps(safe_json_value(row)) + "\n")

                obs = next_obs
                episode_starts = np.array([terminated or truncated], dtype=bool)

                if step % 100 == 0:
                    print(
                        f"[teacher_rollouts] episode={episode} "
                        f"step={step} "
                        f"action={action_name} "
                        f"reward={reward:.4f} "
                        f"total={total_reward:.2f}"
                    )

                if terminated or truncated:
                    print(
                        f"[teacher_rollouts] Episode {episode} ended "
                        f"step={step} total_reward={total_reward:.2f}"
                    )
                    break

        summary_path = episode_dir / "summary.json"
        summary = {
            "episode": episode,
            "total_reward": float(total_reward),
            "metadata_path": str(metadata_path),
            "checkpoint_used": str(CHECKPOINT_PATH) if model is not None else None,
            "visible": visible,
            "max_steps_per_episode": max_steps_per_episode,
        }

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env.close()
    print(f"[teacher_rollouts] Saved rollouts to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    collect_teacher_rollouts(
        episodes=10,
        max_steps_per_episode=1000,
        visible=False,
        use_model=True,
    )
