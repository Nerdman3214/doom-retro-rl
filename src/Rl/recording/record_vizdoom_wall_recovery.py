import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit(f"pygame is required: {e}")

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

    # Try your current ViZDoom env constructor styles.
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
        raise RuntimeError("env.game not found. This recorder needs direct ViZDoom game access.")
    return game


def get_pos(game):
    def gv(name, default=0.0):
        try:
            return float(game.get_game_variable(name))
        except Exception:
            return default

    return {
        "x": gv(vzd.GameVariable.POSITION_X),
        "y": gv(vzd.GameVariable.POSITION_Y),
        "angle": gv(vzd.GameVariable.ANGLE),
        "health": gv(vzd.GameVariable.HEALTH),
        "ammo": gv(vzd.GameVariable.SELECTED_WEAPON_AMMO),
        "kill_count": gv(vzd.GameVariable.KILLCOUNT),
        "item_count": gv(vzd.GameVariable.ITEMCOUNT),
    }


def get_frame(game):
    state = game.get_state()
    if state is None or state.screen_buffer is None:
        return None

    frame = state.screen_buffer

    # ViZDoom usually gives CHW. Convert to HWC.
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[-1] == 3:
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


# These do not require teleporting. They create common stuck states by driving
# into walls/rails/corners from the normal spawn.
TRAP_SETUPS = [
    {
        "name": "half_wall_left_rail_scrape",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 1, 0, 0, 0], 65),
            ([1, 0, 1, 0, 1, 0, 0, 0], 35),
            ([0, 0, 1, 0, 1, 0, 0, 0], 20),
        ],
    },
    {
        "name": "half_wall_right_rail_scrape",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 1, 0, 0], 65),
            ([1, 0, 0, 1, 0, 1, 0, 0], 35),
            ([0, 0, 0, 1, 0, 1, 0, 0], 20),
        ],
    },
    {
        "name": "rail_forward_pressure_left_turn",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 45),
            ([1, 0, 1, 0, 0, 0, 0, 0], 45),
            ([1, 0, 1, 0, 1, 0, 0, 0], 35),
        ],
    },
    {
        "name": "rail_forward_pressure_right_turn",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 45),
            ([1, 0, 0, 1, 0, 0, 0, 0], 45),
            ([1, 0, 0, 1, 0, 1, 0, 0], 35),
        ],
    },

    {
        "name": "front_wall_spawn_push",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 35),
            ([1, 0, 1, 0, 0, 0, 0, 0], 20),
            ([1, 0, 0, 0, 1, 0, 0, 0], 25),
        ],
    },
    {
        "name": "left_wall_slide",
        "sequence": [
            ([1, 0, 0, 0, 1, 0, 0, 0], 45),
            ([1, 0, 1, 0, 1, 0, 0, 0], 25),
        ],
    },
    {
        "name": "right_wall_slide",
        "sequence": [
            ([1, 0, 0, 0, 0, 1, 0, 0], 45),
            ([1, 0, 0, 1, 0, 1, 0, 0], 25),
        ],
    },
    {
        "name": "corner_spin_left",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 30),
            ([0, 0, 1, 0, 0, 0, 0, 0], 25),
            ([1, 0, 1, 0, 0, 0, 0, 0], 40),
        ],
    },
    {
        "name": "corner_spin_right",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 30),
            ([0, 0, 0, 1, 0, 0, 0, 0], 25),
            ([1, 0, 0, 1, 0, 0, 0, 0], 40),
        ],
    },
    {
        "name": "backed_into_bad_angle",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 25),
            ([0, 0, 1, 0, 0, 0, 0, 0], 35),
            ([0, 1, 0, 0, 0, 1, 0, 0], 35),
        ],
    },
]


def reset_and_make_trap(env, game, frame_skip):
    try:
        env.reset()
    except Exception:
        try:
            game.new_episode()
        except Exception:
            pass

    if game.is_episode_finished():
        game.new_episode()

    trap = random.choice(TRAP_SETUPS)

    # Small settle delay.
    run_buttons(game, NO_OP, 5, frame_skip)

    for buttons, steps in trap["sequence"]:
        run_buttons(game, buttons, steps, frame_skip)

    pos = get_pos(game)
    print(
        f"[wall_recovery] trap='{trap['name']}' "
        f"x={pos['x']:.1f} y={pos['y']:.1f} angle={pos['angle']:.1f}"
    )
    print("[wall_recovery] Now recover manually. Press N for next trap, ESC to save/quit.")
    return trap["name"], trap.get("recovery_mode", "wall_recovery")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="vizdoom_wall_recovery_001")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    out_dir = ROOT / "vision_dataset" / "raw_human_play" / args.name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "actions.csv"

    print(f"[wall_recovery] saving to {out_dir}")
    print("[wall_recovery] Controls:")
    print("  W/S/A/D       move/strafe")
    print("  Left/Right    turn")
    print("  Space         shoot")
    print("  E             use")
    print("  N             next random wall trap")
    print("  ESC           stop and save")

    env = make_env()
    game = get_game(env)

    pygame.init()
    pygame.display.set_caption("ViZDoom Wall Recovery Recorder")
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()

    rows = []
    frame_idx = 0
    current_trap, current_recovery_mode = reset_and_make_trap(env, game, args.frame_skip)
    stop = False

    try:
        while not stop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        stop = True
                    elif event.key == pygame.K_n:
                        current_trap, current_recovery_mode = reset_and_make_trap(env, game, args.frame_skip)

            if game.is_episode_finished():
                current_trap, current_recovery_mode = reset_and_make_trap(env, game, args.frame_skip)

            buttons = keys_to_buttons()
            action = buttons_to_action_name(buttons)

            game.make_action(buttons, args.frame_skip)

            frame = get_frame(game)
            frame_name = f"frame_{frame_idx:06d}.png"
            frame_path = frames_dir / frame_name
            save_frame(frame, frame_path)

            pos = get_pos(game)
            rows.append({
                "frame_path": f"frames/{frame_name}",
                "action": action,
                "buttons": " ".join(str(v) for v in buttons),
                "trap": current_trap,
                "recovery_mode": current_recovery_mode,
                "x": pos["x"],
                "y": pos["y"],
                "angle": pos["angle"],
                "health": pos["health"],
                "ammo": pos["ammo"],
                "kill_count": pos["kill_count"],
                "item_count": pos["item_count"],
                "done": bool(game.is_episode_finished()),
                "timestamp": time.time(),
            })

            if frame is not None:
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display = cv2.resize(display, (640, 480))
                pygame.surfarray.blit_array(screen, np.transpose(display, (1, 0, 2)))
                pygame.display.flip()

            if frame_idx % 50 == 0:
                print(
                    f"[wall_recovery] frame={frame_idx} trap={current_trap} "
                    f"action={action} x={pos['x']:.1f} y={pos['y']:.1f}"
                )

            frame_idx += 1
            clock.tick(args.fps)

    finally:
        with csv_path.open("w", newline="") as f:
            fieldnames = [
                "frame_path", "action", "buttons", "trap", "recovery_mode",
                "x", "y", "angle", "health", "ammo",
                "kill_count", "item_count", "done", "timestamp",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        pygame.quit()
        try:
            env.close()
        except Exception:
            pass

        print(f"[wall_recovery] saved rows: {len(rows)}")
        print(f"[wall_recovery] csv: {csv_path}")
        print(f"[wall_recovery] frames: {frames_dir}")


if __name__ == "__main__":
    main()
