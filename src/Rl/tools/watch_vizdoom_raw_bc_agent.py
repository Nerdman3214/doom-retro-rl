from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from env.vizdoom_env import VizDoomEnv


ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "vizdoom_raw_behavior_clone.pt"

DEFAULT_BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]


class SmallCNN(nn.Module):
    def __init__(self, num_buttons: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_buttons),
        )

    def forward(self, x):
        return self.net(x)


def screen_to_tensor(screen) -> torch.Tensor:
    arr = np.asarray(screen)

    # ViZDoom often returns CHW.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    arr = cv2.resize(arr, (84, 84), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def enemy_in_hitscan_lane(state, center_tol: float = 55.0) -> bool:
    """
    Approximate Doom hitscan: enemy must be near the center of the screen.
    This is better than "enemy visible anywhere".
    """
    labels = getattr(state, "labels", None)
    if not labels:
        return False

    # ViZDoom label objects usually have object_name and x/y/width/height.
    for lab in labels:
        name = str(getattr(lab, "object_name", "")).lower()
        value = str(getattr(lab, "value", "")).lower()

        is_enemy = any(tok in name or tok in value for tok in (
            "zombie", "former", "imp", "sergeant", "trooper", "monster", "enemy"
        ))

        if not is_enemy:
            continue

        x = float(getattr(lab, "x", 0.0))
        w = float(getattr(lab, "width", 0.0))
        cx = x + w / 2.0

        # Most ViZDoom screens are 320px wide, but support wider buffers.
        screen_w = 320.0
        try:
            screen_w = float(state.screen_buffer.shape[-1])
        except Exception:
            pass

        if abs(cx - screen_w / 2.0) <= center_tol:
            return True

    return False


def clean_buttons(buttons, probs, names, state):
    idx = {name: i for i, name in enumerate(names)}

    # Never press opposite pairs together. Keep stronger probability.
    for a, b in (("turn_left", "turn_right"), ("strafe_left", "strafe_right"), ("move_forward", "move_backward")):
        if a in idx and b in idx and buttons[idx[a]] and buttons[idx[b]]:
            if probs[idx[a]] >= probs[idx[b]]:
                buttons[idx[b]] = 0
            else:
                buttons[idx[a]] = 0

    # Strong preference: do not move backward unless it is very confident.
    if "move_backward" in idx and buttons[idx["move_backward"]] and probs[idx["move_backward"]] < 0.90:
        buttons[idx["move_backward"]] = 0

    # Hitscan shoot gate: only shoot if enemy is near center aim lane.
    if "shoot" in idx and buttons[idx["shoot"]]:
        if not enemy_in_hitscan_lane(state):
            buttons[idx["shoot"]] = 0

    return buttons


def action_label(buttons, names):
    active = [name for name, val in zip(names, buttons) if val]
    return "+".join(active) if active else "no_op"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skill", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--deterministic-fallback", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT, map_location=device)
    button_names = ckpt.get("button_names", DEFAULT_BUTTON_NAMES)

    model = SmallCNN(num_buttons=len(button_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = VizDoomEnv(visible=True)
    game = getattr(env, "game", None)

    # watch_force_skill
    try:
        game.set_doom_skill(int(args.skill))
        print(f"[viz_bc_watch] set doom skill={args.skill}")
    except Exception as e:
        print(f"[viz_bc_watch] warning: could not set skill: {e}")
    if game is None:
        raise RuntimeError("env.game not found. This watcher needs direct ViZDoom game access.")

    print(f"[viz_bc_watch] loaded {CKPT}")
    print(f"[viz_bc_watch] threshold={args.threshold}")
    print("[viz_bc_watch] watching supervised BC agent. Ctrl+C to stop.")

    ep = 1
    step = 0
    dt = 1.0 / max(args.fps, 1.0)

    if game.is_episode_finished():
        game.new_episode()

    while True:
        if game.is_episode_finished():
            print(f"[viz_bc_watch] episode done ep={ep} steps={step}")
            game.new_episode()
            ep += 1
            step = 0
            time.sleep(0.25)
            continue

        state = game.get_state()
        if state is None:
            game.new_episode()
            continue

        screen = state.screen_buffer
        x = screen_to_tensor(screen).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        buttons = [1 if p >= args.threshold else 0 for p in probs]
        buttons = clean_buttons(buttons, probs, button_names, state)

        # Optional fallback: if completely idle, choose the strongest non-backward action.
        if args.deterministic_fallback and not any(buttons):
            idxs = list(np.argsort(-probs))
            for idx in idxs:
                if button_names[idx] != "move_backward":
                    if probs[idx] >= 0.25:
                        buttons[idx] = 1
                    break

        buttons = clean_buttons(buttons, probs, button_names, state)
        game.make_action(buttons, 1)

        if step % 25 == 0:
            top = sorted(zip(button_names, probs), key=lambda x: x[1], reverse=True)[:4]
            top_str = [(n, round(float(p), 2)) for n, p in top]
            print(
                f"[viz_bc_watch] ep={ep} step={step} "
                f"active={action_label(buttons, button_names)} top={top_str}"
            )

        step += 1
        time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[viz_bc_watch] stopped")
