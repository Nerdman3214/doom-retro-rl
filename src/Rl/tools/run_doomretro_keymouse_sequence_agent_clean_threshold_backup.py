from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pynput import keyboard, mouse
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "doomretro_keymouse_sequence_cnn.pt"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "use",
    "mouse_left",
]

KEY_MAP = {
    "move_forward": "w",
    "move_backward": "s",
    "strafe_left": "a",
    "strafe_right": "d",
    "use": "e",
}

# Clean mouse calibration only.
# Keep Doom Retro in-game mouse sensitivity low.
MOUSE_SCALE = 0.10
MOUSE_DEADZONE = 0.8
MOUSE_MAX_STEP = 1.5
MOUSE_SMOOTHING = 0.12

# Conservative action thresholds, but no recovery/safety systems.
THRESHOLDS = {
    "move_forward": 0.50,
    "move_backward": 0.98,
    "strafe_left": 0.65,
    "strafe_right": 0.65,
    "use": 0.98,
    "mouse_left": 0.90,
}

MAX_RUNTIME_SECONDS = 120.0


class KeyMouseCNN(nn.Module):
    def __init__(self, num_buttons: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )
        self.button_head = nn.Linear(512, num_buttons)
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.button_head(z), self.mouse_head(z)


def get_capture_region():
    with mss.mss() as sct:
        mon = sct.monitors[1]
        return {
            "left": mon["left"],
            "top": mon["top"],
            "width": mon["width"],
            "height": mon["height"],
        }


def simplify_active(probs: torch.Tensor) -> tuple[set[str], dict[str, float]]:
    prob_by_name = {
        name: float(probs[i].item())
        for i, name in enumerate(BUTTON_NAMES)
    }

    active = set()
    for name, threshold in THRESHOLDS.items():
        if prob_by_name.get(name, 0.0) >= threshold:
            active.add(name)

    # Dataset has no keyboard turn labels; mouse handles turning.
    active.discard("turn_left")
    active.discard("turn_right")

    # Do not allow backward in live controller. It caused repeated no_op/backward stalls.
    active.discard("move_backward")

    # Do not hold both strafe directions.
    if "strafe_left" in active and "strafe_right" in active:
        if prob_by_name["strafe_left"] >= prob_by_name["strafe_right"]:
            active.discard("strafe_right")
        else:
            active.discard("strafe_left")

    return active, prob_by_name


def press_button(kb_ctl, mouse_ctl, btn: str):
    if btn == "mouse_left":
        mouse_ctl.press(mouse.Button.left)
        return

    key = KEY_MAP.get(btn)
    if key is not None:
        kb_ctl.press(key)


def release_button(kb_ctl, mouse_ctl, btn: str):
    if btn == "mouse_left":
        mouse_ctl.release(mouse.Button.left)
        return

    key = KEY_MAP.get(btn)
    if key is not None:
        kb_ctl.release(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CKPT))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)

    button_names = ckpt.get("button_names", BUTTON_NAMES)
    if list(button_names) != BUTTON_NAMES:
        print("[seq_agent] warning: checkpoint button_names differ; using checkpoint names")
        globals()["BUTTON_NAMES"][:] = list(button_names)

    model = KeyMouseCNN(num_buttons=len(BUTTON_NAMES)).to(device)

    # Support multiple checkpoint formats from our trainers/backups.
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            print("[seq_agent] checkpoint keys:", sorted(ckpt.keys()))
            raise KeyError("Could not find model weights in checkpoint")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])

    print(f"[seq_agent] loaded: {ckpt_path}")
    print(f"[seq_agent] device: {device}")
    print("[seq_agent] F8 toggle on/off, F9 quit")
    print("[seq_agent] Click/focus Doom Retro, then press F8.")

    region = get_capture_region()
    print(f"[seq_agent] capture: {region}")

    kb_ctl = keyboard.Controller()
    mouse_ctl = mouse.Controller()

    enabled = False
    quitting = False
    enabled_since = None

    frame_buffer = deque(maxlen=4)
    prev_pressed = set()

    smooth_dx = 0.0

    def on_press(key):
        nonlocal enabled, quitting, enabled_since
        if key == keyboard.Key.f8:
            enabled = not enabled
            enabled_since = time.time() if enabled else None
            print(f"[seq_agent] enabled={enabled}")
        elif key == keyboard.Key.f9:
            quitting = True
            print("[seq_agent] quitting")
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_idx = 0
    dt = 1.0 / max(1.0, args.fps)

    try:
        with mss.mss() as sct:
            while not quitting:
                loop_start = time.time()

                if enabled and enabled_since is not None:
                    if time.time() - enabled_since >= MAX_RUNTIME_SECONDS:
                        print("[seq_agent] max runtime reached; auto-stopping")
                        break

                shot = sct.grab(region)
                frame = np.array(shot)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

                pil = Image.fromarray(frame_rgb)
                cur_frame = transform(pil)

                frame_buffer.append(cur_frame)
                while len(frame_buffer) < 4:
                    frame_buffer.append(cur_frame)

                stacked = torch.cat(list(frame_buffer), dim=0)
                x = stacked.unsqueeze(0).to(device)

                with torch.no_grad():
                    button_logits, mouse_pred = model(x)
                    probs = torch.sigmoid(button_logits)[0].detach().cpu()
                    mouse_vec = mouse_pred[0].detach().cpu()

                active, prob_by_name = simplify_active(probs)

                dx = float(mouse_vec[0].item() * 80.0)
                dy = float(mouse_vec[1].item() * 80.0)

                dx *= MOUSE_SCALE
                dy = 0.0

                if abs(dx) < MOUSE_DEADZONE:
                    dx = 0.0

                dx = max(-MOUSE_MAX_STEP, min(MOUSE_MAX_STEP, dx))

                smooth_dx = (1.0 - MOUSE_SMOOTHING) * smooth_dx + MOUSE_SMOOTHING * dx
                dx = smooth_dx

                if abs(dx) < 0.35:
                    dx = 0.0

                # Print model intent even while disabled.
                if frame_idx % args.print_every == 0:
                    top = sorted(
                        prob_by_name.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:4]
                    active_name = "+".join(sorted(active)) if active else "no_op"
                    print(
                        f"[seq_agent] enabled={enabled} "
                        f"active={active_name} "
                        f"mouse=({dx:.1f},0.0) "
                        f"top={[(k, round(v, 2)) for k, v in top]}"
                    )

                if enabled:
                    # Release buttons no longer active.
                    for btn in list(prev_pressed):
                        if btn not in active:
                            release_button(kb_ctl, mouse_ctl, btn)
                            prev_pressed.discard(btn)

                    # Press new active buttons.
                    for btn in active:
                        if btn not in prev_pressed:
                            press_button(kb_ctl, mouse_ctl, btn)
                            prev_pressed.add(btn)

                    if dx != 0.0:
                        mouse_ctl.move(dx, 0)
                else:
                    # Ensure no buttons are held when disabled.
                    for btn in list(prev_pressed):
                        release_button(kb_ctl, mouse_ctl, btn)
                    prev_pressed.clear()

                frame_idx += 1

                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        for btn in list(prev_pressed):
            release_button(kb_ctl, mouse_ctl, btn)
        prev_pressed.clear()
        listener.stop()
        print("[seq_agent] stopped cleanly")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[seq_agent] Ctrl+C received; exiting")
