from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from env.vizdoom_env import VizDoomEnv


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "vision_dataset" / "raw_human_play"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

BTN = {name: i for i, name in enumerate(BUTTON_NAMES)}


def buttons(*names: str) -> list[int]:
    out = [0] * len(BUTTON_NAMES)
    for name in names:
        out[BTN[name]] = 1
    return out


def label_from_buttons(vals: list[int]) -> str:
    active = [name for name, val in zip(BUTTON_NAMES, vals) if val]
    return "+".join(active) if active else "no_op"


def save_frame(path: Path, screen_buffer):
    arr = np.asarray(screen_buffer)

    # ViZDoom usually returns CHW.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def build_teacher_script() -> list[tuple[int, list[int], str]]:
    """
    Each tuple is:
      (num_frames, button_vector, note)

    This is a first scripted route. We will tune these timings based on what
    the teacher does visually.
    """

    script = []

    def add(frames: int, action: list[int], note: str):
        script.append((frames, action, note))

    # Spawn exit / initial push.
    add(70, buttons("move_forward"), "leave_spawn")
    add(20, buttons("move_forward", "shoot"), "clear_first_enemy")
    add(25, buttons("move_forward"), "continue_forward")

    # Turn into route / corridor area.
    add(22, buttons("turn_left"), "turn_left_to_corridor")
    add(75, buttons("move_forward"), "enter_corridor")
    add(18, buttons("turn_right"), "straighten_after_corridor")
    add(60, buttons("move_forward"), "corridor_push")

    # Combat / alignment correction.
    add(20, buttons("shoot"), "shoot_if_enemy")
    add(18, buttons("turn_left"), "adjust_left")
    add(45, buttons("move_forward"), "advance")
    add(14, buttons("turn_right"), "adjust_right")
    add(45, buttons("move_forward"), "advance_more")

    # Door/use section.
    add(8, buttons("use"), "use_door_or_switch")
    add(35, buttons("move_forward"), "after_use_push")

    # Longer route continuation. These are intentionally simple so the BC
    # learner gets a stable route prior.
    add(25, buttons("turn_left"), "route_turn_left")
    add(80, buttons("move_forward"), "long_forward")
    add(16, buttons("turn_right"), "route_turn_right")
    add(80, buttons("move_forward"), "long_forward_2")

    # Final interaction attempt.
    add(10, buttons("use"), "final_use")
    add(50, buttons("move_forward"), "final_push")

    return script


def try_set_skill(game, skill: int):
    try:
        game.set_doom_skill(int(skill))
        print(f"[scripted_teacher] set doom skill={skill}")
    except Exception as e:
        print(f"[scripted_teacher] warning: could not set skill={skill}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--skill", type=int, default=1)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Slow visible playback, e.g. 0.02")
    parser.add_argument("--name-prefix", default="scripted_teacher")
    args = parser.parse_args()

    env = VizDoomEnv(visible=args.visible)
    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env.game not found. Need direct ViZDoom game access.")

    script = build_teacher_script()

    for ep in range(1, args.episodes + 1):
        run_name = f"{args.name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{ep:03d}"
        out_dir = OUT_ROOT / run_name
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "actions.csv"

        print("=" * 80)
        print(f"[scripted_teacher] episode {ep}/{args.episodes}")
        print(f"[scripted_teacher] saving to {out_dir}")

        try_set_skill(game, args.skill)
        game.new_episode()

        rows = []
        frame_idx = 0

        for segment_frames, action, note in script:
            for _ in range(segment_frames):
                if game.is_episode_finished():
                    break

                state = game.get_state()
                if state is None:
                    break

                frame_name = f"frame_{frame_idx:06d}.png"
                frame_path = frames_dir / frame_name
                save_frame(frame_path, state.screen_buffer)

                x = y = 0.0
                try:
                    # Not all env versions expose these reliably through this index,
                    # but keeping fields is useful for debugging.
                    x = float(game.get_game_variable(0))
                    y = float(game.get_game_variable(1))
                except Exception:
                    pass

                rows.append({
                    "frame_path": str(Path("frames") / frame_name),
                    "action": label_from_buttons(action),
                    "buttons": str(action),
                    "source": "scripted_vizdoom_teacher",
                    "episode": ep,
                    "frame": frame_idx,
                    "note": note,
                    "x": x,
                    "y": y,
                })

                game.make_action(action, 1)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(
                        f"[scripted_teacher] frame={frame_idx} "
                        f"action={label_from_buttons(action)} note={note}"
                    )

                if args.sleep > 0:
                    time.sleep(args.sleep)

            if game.is_episode_finished():
                break

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
                    "buttons",
                    "source",
                    "episode",
                    "frame",
                    "note",
                    "x",
                    "y",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[scripted_teacher] saved rows={len(rows)} csv={csv_path}")

    print("[scripted_teacher] done")


if __name__ == "__main__":
    main()
