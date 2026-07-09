from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from env.vizdoom_env import VizDoomEnv


ROOT = Path(__file__).resolve().parents[1]

DOOMRETRO_DATA = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"
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


def parse_active(action: str) -> set[str]:
    action = (action or "").strip()
    if not action or action == "no_op":
        return set()
    return {x.strip() for x in action.split("+") if x.strip()}


def row_to_buttons(row: dict, turn_threshold: float) -> tuple[list[int], str]:
    active = parse_active(row.get("keyboard_action", ""))

    try:
        dx = float(row.get("mouse_dx", 0.0))
    except ValueError:
        dx = 0.0

    mouse_left = str(row.get("mouse_left", "0")).strip() in {"1", "1.0", "true", "True"}

    buttons = [0] * len(BUTTON_NAMES)

    if "move_forward" in active:
        buttons[0] = 1
    if "move_backward" in active:
        buttons[1] = 1

    # Convert Doom Retro mouse turning into ViZDoom discrete turn buttons.
    if dx < -turn_threshold:
        buttons[2] = 1
    elif dx > turn_threshold:
        buttons[3] = 1

    if "strafe_left" in active:
        buttons[4] = 1
    if "strafe_right" in active:
        buttons[5] = 1

    if mouse_left or "shoot" in active:
        buttons[6] = 1

    if "use" in active:
        buttons[7] = 1

    label = "+".join(name for name, val in zip(BUTTON_NAMES, buttons) if val) or "no_op"
    return buttons, label


def save_frame(path: Path, screen_buffer):
    arr = np.asarray(screen_buffer)

    # ViZDoom often returns CHW.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # RGB/BGR ambiguity is not critical for training consistency,
    # but OpenCV writes BGR, so convert if likely RGB.
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="Example: doomretro_easiest_keymouse_completion_001")
    parser.add_argument("--end", default=None, help="Example: doomretro_easiest_keymouse_completion_061")
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--turn-threshold", type=float, default=4.0)
    parser.add_argument("--frame-repeat", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=0, help="0 = no limit")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--skill", type=int, default=1, help="Doom skill 1-5. 1=easiest, 3=normal.")
    args = parser.parse_args()

    runs = sorted(DOOMRETRO_DATA.glob("doomretro_easiest_keymouse_completion_*"))

    if args.start:
        runs = [r for r in runs if r.name >= args.start]
    if args.end:
        runs = [r for r in runs if r.name <= args.end]

    selected = []
    skipped = []

    for run in runs:
        csv_path = run / "actions.csv"
        if not csv_path.exists():
            skipped.append((run.name, "no actions.csv"))
            continue

        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        if len(rows) < args.min_rows:
            skipped.append((run.name, f"rows={len(rows)}"))
            continue

        selected.append((run, rows))

    if args.max_runs and args.max_runs > 0:
        selected = selected[:args.max_runs]

    print(f"[convert] selected runs={len(selected)} skipped={len(skipped)}")
    for name, reason in skipped:
        print(f"[convert] skipped {name}: {reason}")

    env = VizDoomEnv(visible=args.visible)
    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env.game not found. Need direct ViZDoom game access.")

    def apply_skill():
        # Some env versions fail to set skill during __init__, so force it here.
        try:
            game.set_doom_skill(int(args.skill))
            print(f"[convert] set doom skill={args.skill}")
        except Exception as e:
            print(f"[convert] warning: could not set doom skill after init: {e}")

    for run_idx, (run, rows) in enumerate(selected, start=1):
        out_dir = OUT_ROOT / f"converted_{run.name}"
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "actions.csv"

        print("=" * 80)
        print(f"[convert] {run_idx}/{len(selected)} {run.name} rows={len(rows)}")
        print(f"[convert] saving to {out_dir}")

        apply_skill()
        game.new_episode()

        saved_rows = []

        for i, row in enumerate(rows):
            if game.is_episode_finished():
                break

            state = game.get_state()
            if state is None:
                break

            frame_name = f"frame_{i:06d}.png"
            frame_path = frames_dir / frame_name
            save_frame(frame_path, state.screen_buffer)

            buttons, label = row_to_buttons(row, args.turn_threshold)

            try:
                x = float(game.get_game_variable(0))
            except Exception:
                x = 0.0
            try:
                y = float(game.get_game_variable(1))
            except Exception:
                y = 0.0

            saved_rows.append({
                "frame_path": str(Path("frames") / frame_name),
                "action": label,
                "buttons": str(buttons),
                "source": "converted_doomretro_keymouse",
                "original_run": run.name,
                "original_index": i,
                "x": x,
                "y": y,
            })

            game.make_action(buttons, args.frame_repeat)

            if i % 100 == 0:
                print(f"[convert] frame={i} action={label} buttons={buttons}")

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
                    "buttons",
                    "source",
                    "original_run",
                    "original_index",
                    "x",
                    "y",
                ],
            )
            writer.writeheader()
            writer.writerows(saved_rows)

        print(f"[convert] saved rows={len(saved_rows)} csv={out_csv}")

    print("[convert] done")


if __name__ == "__main__":
    main()
