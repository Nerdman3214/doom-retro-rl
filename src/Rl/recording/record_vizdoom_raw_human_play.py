import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit(
        "pygame is required for this recorder. Install it with:\n"
        "  pip install pygame\n"
        f"Original error: {e}"
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.vizdoom_env import VizDoomEnv


DATASET_ROOT = ROOT / "vision_dataset" / "raw_human_play"
SESSION_NAME = datetime.now().strftime("session_%Y%m%d_%H%M%S")
OUTPUT_DIR = DATASET_ROOT / SESSION_NAME
FRAME_DIR = OUTPUT_DIR / "frames"
CSV_PATH = OUTPUT_DIR / "actions.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)


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


def get_game_vars(env):
    state = env.game.get_state()
    if state is None:
        return {}

    try:
        return env._get_game_vars(state)
    except Exception:
        return {}


def get_screen(env):
    state = env.game.get_state()
    if state is None or state.screen_buffer is None:
        return None

    frame = np.asarray(state.screen_buffer)

    # ViZDoom can return CHW.
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
        frame = np.transpose(frame[:3], (1, 2, 0))

    return frame


def save_frame(frame, frame_id):
    frame_path = FRAME_DIR / f"frame_{frame_id:06d}.png"

    if frame is None:
        return ""

    if frame.ndim == 3 and frame.shape[-1] == 3:
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(frame_path), frame)

    return str(frame_path.relative_to(OUTPUT_DIR))


def keys_to_buttons(keys):
    """
    ViZDoom button order from your env:
    0 MOVE_FORWARD
    1 MOVE_BACKWARD
    2 TURN_LEFT
    3 TURN_RIGHT
    4 MOVE_LEFT / strafe_left
    5 MOVE_RIGHT / strafe_right
    6 ATTACK / shoot
    7 USE
    """

    buttons = [0] * 8

    if keys[pygame.K_w]:
        buttons[0] = 1
    if keys[pygame.K_s]:
        buttons[1] = 1
    if keys[pygame.K_LEFT]:
        buttons[2] = 1
    if keys[pygame.K_RIGHT]:
        buttons[3] = 1
    if keys[pygame.K_a]:
        buttons[4] = 1
    if keys[pygame.K_d]:
        buttons[5] = 1
    if keys[pygame.K_SPACE]:
        buttons[6] = 1
    if keys[pygame.K_e]:
        buttons[7] = 1

    return buttons


def buttons_to_label(buttons):
    active = [ACTION_NAMES[i] for i, v in enumerate(buttons) if v]

    if not active:
        return "no_op"

    # For supervised learning, keep compound actions.
    # Examples:
    #   move_forward+turn_left
    #   strafe_right+shoot
    return "+".join(active)


def main():
    print("[raw_record] Starting raw human gameplay recorder.")
    print("")
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
    print("")
    print("This recorder bypasses env.step(), sanitize_action(), and curriculum.")
    print("It sends raw ViZDoom button vectors directly with game.make_action().")
    print("")

    # visible=False because pygame will show the gameplay frame.
    # This avoids keyboard focus fighting with the ViZDoom window.
    env = VizDoomEnv(visible=False)
    env.reset()

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("ViZDoom Human Recorder - click here, ESC to stop")
    clock = pygame.time.Clock()

    rows = []
    frame_id = 0
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            buttons = keys_to_buttons(keys)
            action_label = buttons_to_label(buttons)

            # Direct raw control. No RL env action sanitizer.
            env.game.make_action(buttons, env.frame_skip)

            done = env.game.is_episode_finished()
            truncated = env.game.get_episode_time() >= env.max_episode_steps * env.frame_skip

            frame = get_screen(env)
            frame_path = save_frame(frame, frame_id)
            vars_now = get_game_vars(env)

            rows.append({
                "frame_path": frame_path,
                "action": action_label,
                "buttons": " ".join(str(x) for x in buttons),
                "x": vars_now.get("x"),
                "y": vars_now.get("y"),
                "angle": vars_now.get("angle"),
                "health": vars_now.get("health"),
                "ammo": vars_now.get("ammo"),
                "kill_count": vars_now.get("kill_count"),
                "item_count": vars_now.get("item_count"),
                "done": bool(done),
                "truncated": bool(truncated),
                "timestamp": time.time(),
            })

            if frame is not None:
                # Show frame in pygame.
                display = frame
                if display.ndim == 3 and display.shape[-1] == 3:
                    display = cv2.resize(display, (640, 480))
                    surface = pygame.surfarray.make_surface(np.transpose(display, (1, 0, 2)))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()

            if frame_id % 50 == 0:
                print(
                    f"[raw_record] frame={frame_id} "
                    f"action={action_label} "
                    f"buttons={buttons} "
                    f"x={vars_now.get('x')} y={vars_now.get('y')}"
                )

            frame_id += 1

            if done or truncated:
                env.reset()

            # 20 FPS-ish recording.
            clock.tick(20)

    except KeyboardInterrupt:
        print("")
        print("[raw_record] Ctrl+C received. Saving dataset...")

    finally:
        pygame.quit()
        env.close()

        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_path",
                    "action",
                    "buttons",
                    "x",
                    "y",
                    "angle",
                    "health",
                    "ammo",
                    "kill_count",
                    "item_count",
                    "done",
                    "truncated",
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print("")
        print(f"[raw_record] saved rows: {len(rows)}")
        print(f"[raw_record] csv: {CSV_PATH}")
        print(f"[raw_record] frames: {FRAME_DIR}")


if __name__ == "__main__":
    main()
