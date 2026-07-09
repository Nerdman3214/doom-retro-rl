#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import os
import time

import numpy as np
from PIL import Image

import pygame
import vizdoom as vzd


BUTTON_ORDER = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
]

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


def action_name(buttons):
    names = [ACTION_NAMES[i] for i, v in enumerate(buttons) if int(v)]
    return "+".join(names) if names else "no_op"


def screen_to_hwc(screen):
    arr = np.asarray(screen)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    return arr.astype(np.uint8)


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    # Exact order used below.
    for name in [
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "ANGLE",
        "HEALTH",
        "AMMO2",
        "KILLCOUNT",
        "ITEMCOUNT",
    ]:
        if hasattr(vzd.GameVariable, name):
            game.add_available_game_variable(getattr(vzd.GameVariable, name))

    return game


def pressed_to_buttons(keys):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iwad", default=os.environ.get("VIZDOOM_IWAD", "/usr/share/games/doom/freedoom1.wad"))
    ap.add_argument("--map", default=os.environ.get("VIZDOOM_MAP", "E1M1"))
    ap.add_argument("--skill", type=int, default=int(os.environ.get("VIZDOOM_SKILL", "1")))
    ap.add_argument("--fps", type=float, default=35.0)
    ap.add_argument("--episode-timeout", type=int, default=8000)
    ap.add_argument("--out-root", default="vision_dataset/raw_human_play")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = Path(args.out_root) / f"session_z_{stamp}"
    frames_dir = session / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path = session / "actions.csv"

    game = make_game(args)
    game.init()
    game.new_episode()

    pygame.init()
    display = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("ViZDoom raw human recorder with z/elevation")
    clock = pygame.time.Clock()

    print("[raw_record_z] Starting z-aware recorder.")
    print("[raw_record_z] session:", session)
    print("[raw_record_z] iwad:", args.iwad)
    print("[raw_record_z] map:", args.map)
    print("[raw_record_z] skill:", args.skill)
    print()
    print("Controls:")
    print("  W      move_forward")
    print("  S      move_backward")
    print("  A      strafe_left")
    print("  D      strafe_right")
    print("  Left   turn_left")
    print("  Right  turn_right")
    print("  Space  shoot")
    print("  E      use")
    print("  ESC    stop and save")
    print()

    fieldnames = [
        "frame_path",
        "frame",
        "action",
        "buttons",
        "x",
        "y",
        "z",
        "angle",
        "health",
        "ammo",
        "kill_count",
        "item_count",
        "done",
        "truncated",
        "timestamp",
    ]

    rows = []
    frame_id = 0
    stopped = False

    try:
        while not game.is_episode_finished() and not stopped:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stopped = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    stopped = True

            state = game.get_state()
            if state is None:
                break

            screen = screen_to_hwc(state.screen_buffer)

            # Show frame in pygame.
            surf = pygame.surfarray.make_surface(np.transpose(screen, (1, 0, 2)))
            display.blit(surf, (0, 0))
            pygame.display.flip()

            keys = pygame.key.get_pressed()
            buttons = pressed_to_buttons(keys)
            action = action_name(buttons)

            vars_now = list(state.game_variables)
            # Expected variable order:
            # x, y, z, angle, health, ammo, kill_count, item_count
            x = vars_now[0] if len(vars_now) > 0 else 0.0
            y = vars_now[1] if len(vars_now) > 1 else 0.0
            z = vars_now[2] if len(vars_now) > 2 else 0.0
            angle = vars_now[3] if len(vars_now) > 3 else 0.0
            health = vars_now[4] if len(vars_now) > 4 else 0.0
            ammo = vars_now[5] if len(vars_now) > 5 else 0.0
            kill_count = vars_now[6] if len(vars_now) > 6 else 0.0
            item_count = vars_now[7] if len(vars_now) > 7 else 0.0

            rel_frame = f"frames/frame_{frame_id:06d}.png"
            frame_path = session / rel_frame
            Image.fromarray(screen).save(frame_path)

            rows.append({
                "frame_path": rel_frame,
                "frame": frame_id,
                "action": action,
                "buttons": str(buttons),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "angle": float(angle),
                "health": float(health),
                "ammo": float(ammo),
                "kill_count": float(kill_count),
                "item_count": float(item_count),
                "done": False,
                "truncated": False,
                "timestamp": time.time(),
            })

            if frame_id % 50 == 0:
                print(
                    f"[raw_record_z] frame={frame_id} action={action} "
                    f"x={float(x):.1f} y={float(y):.1f} z={float(z):.1f} angle={float(angle):.1f}"
                )

            game.make_action(buttons, 1)

            frame_id += 1
            clock.tick(args.fps)

    except KeyboardInterrupt:
        print("[raw_record_z] stopped by keyboard")

    finally:
        if rows:
            rows[-1]["done"] = bool(game.is_episode_finished())

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        game.close()
        pygame.quit()

    print()
    print("[raw_record_z] saved rows:", len(rows))
    print("[raw_record_z] csv:", csv_path)
    print("[raw_record_z] frames:", frames_dir)


if __name__ == "__main__":
    main()
