#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import argparse
import time

import numpy as np
from PIL import Image

import torch
import vizdoom as vzd

from training.train_vizdoom_video_behavior_clone import VideoBCNet


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

DEFAULT_ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

ACTION_TO_BUTTON = {
    "move_forward": "MOVE_FORWARD",
    "move_backward": "MOVE_BACKWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    "strafe_left": "MOVE_LEFT",
    "strafe_right": "MOVE_RIGHT",
    "shoot": "ATTACK",
    "use": "USE",
}


def screen_to_tensor(screen_buffer, image_size):
    arr = np.asarray(screen_buffer)

    # Handle CHW from ViZDoom if needed.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(args.iwad)
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_window_visible(args.visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(args.episode_timeout)

    for name in BUTTON_ORDER:
        game.add_available_button(getattr(vzd.Button, name))

    for var in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "POSITION_Z"]:
        if hasattr(vzd.GameVariable, var):
            game.add_available_game_variable(getattr(vzd.GameVariable, var))

    return game


def resolve_conflicts(probs, action_names, active):
    # Do not press opposite direction pairs at the same time.
    opposite_pairs = [
        ("turn_left", "turn_right"),
        ("strafe_left", "strafe_right"),
        ("move_forward", "move_backward"),
    ]

    active = set(active)
    prob_map = {name: float(probs[i]) for i, name in enumerate(action_names)}

    for a, b in opposite_pairs:
        if a in active and b in active:
            if prob_map.get(a, 0.0) >= prob_map.get(b, 0.0):
                active.remove(b)
            else:
                active.remove(a)

    return sorted(active)


def actions_to_buttons(active):
    active_buttons = set(ACTION_TO_BUTTON[a] for a in active if a in ACTION_TO_BUTTON)
    return [1 if name in active_buttons else 0 for name in BUTTON_ORDER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vizdoom_video_behavior_clone.pt")
    ap.add_argument("--iwad", default="/usr/share/games/doom/freedoom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1)
    ap.add_argument("--visible", action="store_true", default=True)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--fallback-top", action="store_true")
    ap.add_argument("--fallback-min", type=float, default=0.10)
    ap.add_argument("--episode-timeout", type=int, default=4000)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    seq_len = int(ckpt.get("seq_len", 16))
    image_size = int(ckpt.get("image_size", 128))
    action_names = list(ckpt.get("action_names", DEFAULT_ACTION_NAMES))

    model = VideoBCNet(num_actions=len(action_names)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    game = make_game(args)

    print("[video_watch] checkpoint:", ckpt_path)
    print("[video_watch] device:", device)
    print("[video_watch] seq_len:", seq_len)
    print("[video_watch] image_size:", image_size)
    print("[video_watch] threshold:", args.threshold)
    print("[video_watch] fallback_top:", args.fallback_top)
    print("[video_watch] actions:", action_names)
    print("[video_watch] buttons:", BUTTON_ORDER)

    game.init()
    frame_dt = 1.0 / max(1.0, args.fps)

    try:
        for ep in range(1, args.repeat + 1):
            game.new_episode()
            history = deque(maxlen=seq_len)

            print(f"[video_watch] episode {ep}/{args.repeat}")

            # Warm up history by repeating the initial screen.
            state = game.get_state()
            if state is None:
                continue

            first = screen_to_tensor(state.screen_buffer, image_size)
            for _ in range(seq_len):
                history.append(first)

            step = 0

            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break

                history.append(screen_to_tensor(state.screen_buffer, image_size))

                x = torch.stack(list(history), dim=0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

                active = [
                    action_names[i]
                    for i, p in enumerate(probs)
                    if float(p) >= args.threshold
                ]

                if not active and args.fallback_top:
                    top_i = int(np.argmax(probs))
                    if float(probs[top_i]) >= args.fallback_min:
                        active = [action_names[top_i]]

                active = resolve_conflicts(probs, action_names, active)
                buttons = actions_to_buttons(active)

                reward = game.make_action(buttons, 1)

                if step % 25 == 0:
                    top = sorted(
                        [(action_names[i], round(float(probs[i]), 3)) for i in range(len(action_names))],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                    vars_now = list(state.game_variables) if state.game_variables is not None else []
                    print(
                        f"[video_watch] ep={ep} step={step} "
                        f"active={'+'.join(active) if active else 'no_op'} "
                        f"top={top} reward={reward} vars={vars_now}"
                    )

                step += 1

                if args.visible:
                    time.sleep(frame_dt)

            print(f"[video_watch] episode done ep={ep} steps={step}")

    except KeyboardInterrupt:
        print("[video_watch] stopped by user")

    finally:
        game.close()

    print("[video_watch] done")


if __name__ == "__main__":
    main()
