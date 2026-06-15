"""
Generate many ViZDoom wall / rail / half-wall recovery situations.

Goal:
- Create 1000+ randomized stuck-like situations.
- Save frames + metadata.
- Let the human optionally record recovery actions for good stuck cases.

Controls during human recovery:
  W/S/A/D       move/strafe
  Left/Right    turn
  Space         shoot
  E             use
  N             next situation
  R             start/stop recording recovery for current situation
  ESC           save and quit

This file creates a dataset at:
  vision_dataset/recovery_situations/<name>/
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

try:
    import vizdoom as vzd
except Exception as e:
    raise SystemExit(f"vizdoom is required: {e}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import vizdoom_env as viz_env_module


ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

NO_OP = [0, 0, 0, 0, 0, 0, 0, 0]


def buttons_to_action_name(buttons):
    active = [name for name, value in zip(ACTION_NAMES, buttons) if value]
    return "+".join(active) if active else "no_op"


def keys_to_buttons():
    keys = pygame.key.get_pressed()
    return [
        int(keys[pygame.K_w]),
        int(keys[pygame.K_s]),
        int(keys[pygame.K_LEFT]),
        int(keys[pygame.K_RIGHT]),
        int(keys[pygame.K_a]),
        int(keys[pygame.K_d]),
        int(keys[pygame.K_SPACE]),
        int(keys[pygame.K_e]),
    ]


def make_env():
    EnvClass = getattr(viz_env_module, "VizDoomEnv", None) or getattr(viz_env_module, "DoomEnv", None)
    if EnvClass is None:
        raise RuntimeError("Could not find VizDoomEnv or DoomEnv in env/vizdoom_env.py")

    for kwargs in [
        {"visible": False, "depth": True, "labels": True, "automap": True},
        {"visible": False},
        {},
    ]:
        try:
            return EnvClass(**kwargs)
        except TypeError:
            continue

    return EnvClass()


def get_game(env):
    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env.game not found. Need direct ViZDoom game access.")
    return game


def gv(game, variable, default=0.0):
    try:
        return float(game.get_game_variable(variable))
    except Exception:
        return default


def get_pos(game):
    return {
        "x": gv(game, vzd.GameVariable.POSITION_X),
        "y": gv(game, vzd.GameVariable.POSITION_Y),
        "angle": gv(game, vzd.GameVariable.ANGLE),
        "health": gv(game, vzd.GameVariable.HEALTH),
        "ammo": gv(game, vzd.GameVariable.SELECTED_WEAPON_AMMO),
        "kill_count": gv(game, vzd.GameVariable.KILLCOUNT),
        "item_count": gv(game, vzd.GameVariable.ITEMCOUNT),
    }


def get_frame(game):
    state = game.get_state()
    if state is None or state.screen_buffer is None:
        return None

    frame = state.screen_buffer

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def save_frame(frame, path):
    if frame is None:
        return False
    cv2.imwrite(str(path), frame)
    return True


def run_buttons(game, buttons, steps, frame_skip):
    for _ in range(steps):
        if game.is_episode_finished():
            break
        game.make_action(buttons, frame_skip)


def dist(a, b):
    return float(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5)


# Randomized trap templates.
# These intentionally create bad orientation + rail/wall pressure states.
TRAP_TEMPLATES = [
    {
        "trap_family": "left_wall_slide",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 1, 0, 0, 0], (30, 80)),
            ([1, 0, 1, 0, 1, 0, 0, 0], (10, 55)),
        ],
    },
    {
        "trap_family": "right_wall_slide",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 0, 1, 0, 0], (30, 80)),
            ([1, 0, 0, 1, 0, 1, 0, 0], (10, 55)),
        ],
    },
    {
        "trap_family": "front_wall_pressure",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (40, 110)),
            ([1, 0, 1, 0, 0, 0, 0, 0], (0, 35)),
        ],
    },
    {
        "trap_family": "corner_left_spin",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (20, 70)),
            ([0, 0, 1, 0, 0, 0, 0, 0], (20, 70)),
            ([1, 0, 1, 0, 0, 0, 0, 0], (20, 75)),
        ],
    },
    {
        "trap_family": "corner_right_spin",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (20, 70)),
            ([0, 0, 0, 1, 0, 0, 0, 0], (20, 70)),
            ([1, 0, 0, 1, 0, 0, 0, 0], (20, 75)),
        ],
    },
    {
        "trap_family": "half_wall_left_rail",
        "recovery_mode": "half_wall_rail",
        "base": [
            ([1, 0, 0, 0, 1, 0, 0, 0], (60, 140)),
            ([1, 0, 1, 0, 1, 0, 0, 0], (20, 80)),
            ([0, 0, 1, 0, 1, 0, 0, 0], (10, 50)),
        ],
    },
    {
        "trap_family": "half_wall_right_rail",
        "recovery_mode": "half_wall_rail",
        "base": [
            ([1, 0, 0, 0, 0, 1, 0, 0], (60, 140)),
            ([1, 0, 0, 1, 0, 1, 0, 0], (20, 80)),
            ([0, 0, 0, 1, 0, 1, 0, 0], (10, 50)),
        ],
    },
    {
        "trap_family": "rail_forward_left_turn",
        "recovery_mode": "half_wall_rail",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (40, 110)),
            ([1, 0, 1, 0, 0, 0, 0, 0], (40, 110)),
            ([1, 0, 1, 0, 1, 0, 0, 0], (20, 90)),
        ],
    },
    {
        "trap_family": "rail_forward_right_turn",
        "recovery_mode": "half_wall_rail",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (40, 110)),
            ([1, 0, 0, 1, 0, 0, 0, 0], (40, 110)),
            ([1, 0, 0, 1, 0, 1, 0, 0], (20, 90)),
        ],
    },
    {
        "trap_family": "bad_reverse_angle",
        "recovery_mode": "wall_recovery",
        "base": [
            ([1, 0, 0, 0, 0, 0, 0, 0], (20, 80)),
            ([0, 0, 1, 0, 0, 0, 0, 0], (30, 100)),
            ([0, 1, 0, 0, 0, 1, 0, 0], (20, 90)),
        ],
    },
]


def make_random_sequence(template):
    sequence = []

    # Random initial rotation to diversify angle.
    pre_turn_choice = random.choice([
        NO_OP,
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ])
    pre_turn_steps = random.randint(0, 80)
    if pre_turn_steps > 0:
        sequence.append((pre_turn_choice, pre_turn_steps))

    for buttons, step_range in template["base"]:
        lo, hi = step_range
        steps = random.randint(lo, hi)

        noisy = list(buttons)

        # Randomly add/remove forward pressure to diversify.
        if random.random() < 0.15:
            noisy[0] = 1

        # Occasionally add tiny opposite strafe noise.
        if random.random() < 0.10:
            noisy[4] = int(not noisy[4])
        if random.random() < 0.10:
            noisy[5] = int(not noisy[5])

        sequence.append((noisy, steps))

    return sequence


def reset_game(env, game):
    try:
        env.reset()
    except Exception:
        try:
            game.new_episode()
        except Exception:
            pass

    if game.is_episode_finished():
        game.new_episode()


def generate_situation(env, game, frame_skip):
    reset_game(env, game)
    run_buttons(game, NO_OP, random.randint(2, 10), frame_skip)

    template = random.choice(TRAP_TEMPLATES)
    sequence = make_random_sequence(template)

    start_pos = get_pos(game)

    for buttons, steps in sequence:
        run_buttons(game, buttons, steps, frame_skip)

    end_pos = get_pos(game)
    frame = get_frame(game)

    situation = {
        "trap_family": template["trap_family"],
        "recovery_mode": template["recovery_mode"],
        "start_x": start_pos["x"],
        "start_y": start_pos["y"],
        "start_angle": start_pos["angle"],
        "x": end_pos["x"],
        "y": end_pos["y"],
        "angle": end_pos["angle"],
        "health": end_pos["health"],
        "ammo": end_pos["ammo"],
        "setup_distance": dist(start_pos, end_pos),
        "sequence_len": sum(steps for _, steps in sequence),
    }

    return situation, frame


def draw_text(frame, lines):
    if frame is None:
        return None
    img = frame.copy()
    y = 24
    for line in lines:
        cv2.putText(
            img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="recovery_situations_001")
    parser.add_argument("--num-situations", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--human", action="store_true", help="Allow manual recording after each situation.")
    parser.add_argument("--auto-only", action="store_true", help="Only generate situations; no human UI loop.")
    args = parser.parse_args()

    out_dir = ROOT / "vision_dataset" / "recovery_situations" / args.name
    situation_frames = out_dir / "situation_frames"
    recovery_frames = out_dir / "recovery_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    situation_frames.mkdir(parents=True, exist_ok=True)
    recovery_frames.mkdir(parents=True, exist_ok=True)

    situations_csv = out_dir / "situations.csv"
    recoveries_csv = out_dir / "recoveries.csv"

    print(f"[recovery_situations] saving to: {out_dir}")
    print(f"[recovery_situations] num_situations={args.num_situations}")

    env = make_env()
    game = get_game(env)

    situation_rows = []
    recovery_rows = []

    if args.auto_only:
        for i in range(args.num_situations):
            situation, frame = generate_situation(env, game, args.frame_skip)
            frame_name = f"situation_{i:06d}.png"
            save_frame(frame, situation_frames / frame_name)

            situation["situation_id"] = i
            situation["frame_path"] = f"situation_frames/{frame_name}"
            situation_rows.append(situation)

            if i % 50 == 0:
                print(
                    f"[recovery_situations] {i}/{args.num_situations} "
                    f"{situation['trap_family']} mode={situation['recovery_mode']} "
                    f"x={situation['x']:.1f} y={situation['y']:.1f}"
                )

        with situations_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(situation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(situation_rows)

        print(f"[recovery_situations] wrote {situations_csv}")
        return

    pygame.init()
    pygame.display.set_caption("ViZDoom Recovery Situation Generator")
    screen = pygame.display.set_mode((960, 720))
    clock = pygame.time.Clock()

    current_id = 0
    frame_idx = 0
    recording = False
    stop = False

    situation, frame = generate_situation(env, game, args.frame_skip)

    print("[recovery_situations] Controls:")
    print("  R = start/stop recording recovery")
    print("  N = next situation")
    print("  ESC = save and quit")
    print("  W/S/A/D, arrows, Space, E = gameplay controls")

    while not stop and current_id < args.num_situations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    stop = True

                elif event.key == pygame.K_n:
                    # Save situation snapshot before moving on.
                    frame_name = f"situation_{current_id:06d}.png"
                    save_frame(frame, situation_frames / frame_name)

                    row = dict(situation)
                    row["situation_id"] = current_id
                    row["frame_path"] = f"situation_frames/{frame_name}"
                    situation_rows.append(row)

                    current_id += 1
                    frame_idx = 0
                    recording = False

                    if current_id < args.num_situations:
                        situation, frame = generate_situation(env, game, args.frame_skip)
                        print(
                            f"[recovery_situations] new situation {current_id}: "
                            f"{situation['trap_family']} mode={situation['recovery_mode']} "
                            f"x={situation['x']:.1f} y={situation['y']:.1f} angle={situation['angle']:.1f}"
                        )

                elif event.key == pygame.K_r:
                    recording = not recording
                    print(f"[recovery_situations] recording={recording}")

        buttons = keys_to_buttons()
        action = buttons_to_action_name(buttons)

        if not game.is_episode_finished():
            game.make_action(buttons, args.frame_skip)

        live_frame = get_frame(game)
        if live_frame is not None:
            frame = live_frame

        pos = get_pos(game)

        if recording:
            frame_name = f"situation_{current_id:06d}_frame_{frame_idx:06d}.png"
            save_frame(frame, recovery_frames / frame_name)

            recovery_rows.append({
                "situation_id": current_id,
                "frame_path": f"recovery_frames/{frame_name}",
                "action": action,
                "buttons": " ".join(str(v) for v in buttons),
                "trap_family": situation["trap_family"],
                "recovery_mode": situation["recovery_mode"],
                "x": pos["x"],
                "y": pos["y"],
                "angle": pos["angle"],
                "health": pos["health"],
                "ammo": pos["ammo"],
                "kill_count": pos["kill_count"],
                "item_count": pos["item_count"],
                "timestamp": time.time(),
            })
            frame_idx += 1

        display = draw_text(frame, [
            f"id={current_id}/{args.num_situations}",
            f"trap={situation['trap_family']}",
            f"mode={situation['recovery_mode']}",
            f"recording={recording}",
            "R record | N next | ESC save",
        ])

        if display is not None:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (960, 720))
            pygame.surfarray.blit_array(screen, np.transpose(display, (1, 0, 2)))
            pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()

    if situation_rows:
        with situations_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(situation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(situation_rows)

    if recovery_rows:
        with recoveries_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(recovery_rows[0].keys()))
            writer.writeheader()
            writer.writerows(recovery_rows)

    try:
        env.close()
    except Exception:
        pass

    print(f"[recovery_situations] situations: {len(situation_rows)} -> {situations_csv}")
    print(f"[recovery_situations] recovery rows: {len(recovery_rows)} -> {recoveries_csv}")


if __name__ == "__main__":
    main()
