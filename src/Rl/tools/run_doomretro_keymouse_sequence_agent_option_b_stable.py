"""
Live Doom Retro imitation agent.

Controls:
  F8 = toggle agent control on/off
  F9 = quit immediately

Before running:
  1. Start Doom Retro manually.
  2. Load E1M1 on easiest.
  3. Click/focus the Doom Retro window.
  4. Run this script.
  5. Press F8 to let the model control.
  6. Press F8 again to stop control.
  7. Press F9 to quit.

This uses:
  checkpoints/doomretro_keymouse_imitation_cnn.pt
"""

from pathlib import Path
import time
from collections import deque
from threading import Lock

import cv2
import mss
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T

from pynput import keyboard, mouse


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
    "use": keyboard.Key.space,
}

# Conservative thresholds for live control.
THRESHOLDS = {
    "move_forward": 0.50,

    # Disabled for now. The model overpredicts backing up live.
    # We can bring this back later for wall recovery.
    "move_backward": 0.98,

    "strafe_left": 0.65,
    "strafe_right": 0.65,
    "use": 0.98,
    "mouse_left": 0.90,
}

MOUSE_SCALE = 0.10
MOUSE_DEADZONE = 0.8
MOUSE_MAX_STEP = 1.5

# Smooth mouse turns so the agent rotates gradually instead of snapping.
MOUSE_SMOOTHING = 0.12

# Send mouse movement only occasionally.
# This prevents tiny movements from accumulating too fast.
MOUSE_PULSE_EVERY_N_FRAMES = 1

# Route/recovery layer.
# This sits above imitation and prevents wall/no-op loops.
ROUTE_RECOVERY_FRAMES = 8
NOOP_STUCK_LIMIT = 8
LOW_PROGRESS_LIMIT = 12
TURN_LOCK_LIMIT = 10
RECOVERY_MOUSE_STEP = 0.0

# Safety: auto-exit so the script cannot run forever.
MAX_RUNTIME_SECONDS = 120.0

running = True
agent_enabled = False
state_lock = Lock()


class KeyMouseCNN(nn.Module):
    def __init__(self, num_buttons):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.button_head = nn.Linear(512, num_buttons)
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.button_head(z), self.mouse_head(z)


def on_press(key):
    global running, agent_enabled

    if key == keyboard.Key.f8:
        with state_lock:
            agent_enabled = not agent_enabled
            print(f"[seq_agent] enabled={agent_enabled}")

    if key == keyboard.Key.f9:
        with state_lock:
            running = False
            agent_enabled = False
            print("[seq_agent] quitting")


def release_all(kbd, mse):
    for action, key in KEY_MAP.items():
        try:
            kbd.release(key)
        except Exception:
            pass

    try:
        mse.release(mouse.Button.left)
    except Exception:
        pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    if not CKPT.exists():
        raise SystemExit(f"Missing checkpoint: {CKPT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CKPT, map_location=device)
    button_names = ckpt.get("button_names", BUTTON_NAMES)
    mouse_scale = float(ckpt.get("mouse_scale", 80.0))

    model = KeyMouseCNN(num_buttons=len(button_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((84, 84)),
        T.ToTensor(),
    ])

    kbd = keyboard.Controller()
    mse = mouse.Controller()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("[seq_agent] loaded:", CKPT)
    print("[seq_agent] device:", device)
    print("[seq_agent] F8 toggle on/off, F9 quit")
    print("[seq_agent] Click/focus Doom Retro, then press F8.")

    prev_pressed = set()
    last_active_signature = None
    same_action_frames = 0
    frame_idx = 0
    frame_buffer = deque(maxlen=4)
    smooth_dx = 0.0
    smooth_dy = 0.0
    soft_use_frames = 0
    no_op_frames = 0

    # Route/recovery state.
    recovery_frames = 0
    recovery_cooldown = 0
    recovery_side = 1
    low_progress_frames = 0
    repeated_turn_frames = 0
    last_dx_sign = 0
    recent_active = deque(maxlen=16)
    agent_start_time = time.time()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        print("[seq_agent] capture:", monitor)

        while True:
            with state_lock:
                local_running = running
                local_enabled = agent_enabled

            if time.time() - agent_start_time >= MAX_RUNTIME_SECONDS:
                print("[seq_agent] max runtime reached; auto-stopping")
                break

            if not local_running:
                break

            start = time.time()

            img = np.array(sct.grab(monitor))
            frame_rgb = img[:, :, :3]

            pil = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
            cur_frame = transform(pil)

            frame_buffer.append(cur_frame)

            while len(frame_buffer) < 4:
                frame_buffer.append(cur_frame)

            stacked = torch.cat(list(frame_buffer), dim=0)
            x = stacked.unsqueeze(0).to(device)

            with torch.no_grad():
                button_logits, mouse_pred = model(x)
                probs = torch.sigmoid(button_logits)[0].detach().cpu()
                m = mouse_pred[0].detach().cpu()

            active = set()

            prob_by_name = {
                name: float(probs[i])
                for i, name in enumerate(button_names)
            }

            for name, threshold in THRESHOLDS.items():
                if prob_by_name.get(name, 0.0) >= threshold:
                    active.add(name)

            # Remove contradictory movement.
            if "move_forward" in active and "move_backward" in active:
                if prob_by_name.get("move_forward", 0.0) >= prob_by_name.get("move_backward", 0.0):
                    active.discard("move_backward")
                else:
                    active.discard("move_forward")

            # Safety rule for the first live version:
            # do not let the agent backpedal around the map.
            # Backward movement is useful later for stuck recovery, but bad for route completion.
            active.discard("move_backward")

            if "strafe_left" in active and "strafe_right" in active:
                if prob_by_name.get("strafe_left", 0.0) >= prob_by_name.get("strafe_right", 0.0):
                    active.discard("strafe_right")
                else:
                    active.discard("strafe_left")

            # Suppress weak mouse shooting in live control.
            # Offline it learned mouse_left, but live it is too eager.
            if prob_by_name.get("mouse_left", 0.0) < 0.90:
                active.discard("mouse_left")

            raw_dx = float(m[0]) * mouse_scale
            raw_dy = float(m[1]) * mouse_scale

            dx = raw_dx * MOUSE_SCALE
            dy = raw_dy * MOUSE_SCALE

            if abs(dx) < MOUSE_DEADZONE:
                dx = 0.0
            if abs(dy) < MOUSE_DEADZONE:
                dy = 0.0

            dx = clamp(dx, -MOUSE_MAX_STEP, MOUSE_MAX_STEP)
            dy = clamp(dy, -MOUSE_MAX_STEP, MOUSE_MAX_STEP)

            # Anti-strafe-lock safety:
            # If the agent keeps choosing only strafe for too long,
            # force a forward bias so it keeps trying to progress.
            active_signature = "+".join(sorted(active)) if active else "no_op"
            if active_signature == last_active_signature:
                same_action_frames += 1
            else:
                same_action_frames = 0
                last_active_signature = active_signature

            strafe_only = (
                ("strafe_left" in active or "strafe_right" in active)
                and "move_forward" not in active
                and "mouse_left" not in active
                and "use" not in active
            )

            if strafe_only and same_action_frames >= 8:
                active.discard("strafe_left")
                active.discard("strafe_right")
                active.add("move_forward")

            # Avoid use while strafing. Use should be deliberate.
            if "use" in active and ("strafe_left" in active or "strafe_right" in active):
                active.discard("use")

            # Anti-use-spam safety:
            # only allow use when the model is extremely confident.
            if prob_by_name.get("use", 0.0) < 0.98:
                active.discard("use")

            # LOWEST_MOUSE_PROFILE_FINAL_CLAMP
            dx = max(-1.5, min(1.5, dx))
            dy = 0.0  # HARD_VERTICAL_MOUSE_OFF

            if abs(dx) < 0.35:
                dx = 0.0

            # ULTRA_SLOW_MOUSE_PULSE_GATE
            # Even dx=1 can turn too fast in Doom Retro, so only send
            # mouse movement once every few frames.
            if frame_idx % MOUSE_PULSE_EVERY_N_FRAMES != 0:
                dx = 0.0
                dy = 0.0

            # SOFT_USE_AND_NOOP_RECOVERY
            # If the model keeps predicting use around 0.85-0.97 but safety blocks it,
            # allow a short tap instead of freezing forever.
            if prob_by_name.get("use", 0.0) >= 0.85 and "use" not in active:
                soft_use_frames += 1
            else:
                soft_use_frames = 0

            if not active:
                no_op_frames += 1
            else:
                no_op_frames = 0

            # Tap use once after repeated soft-use confidence.
            if soft_use_frames >= 8:
                active.add("use")
                soft_use_frames = 0

            # If the agent freezes too long, bias it back into moving.
            if no_op_frames >= 8:
                active.add("move_forward")
                no_op_frames = 0

            # ROUTE_RECOVERY_LAYER
            if recovery_cooldown > 0:
                recovery_cooldown -= 1

            # Track recent action signatures.
            recent_active.append("+".join(sorted(active)) if active else "no_op")

            dx_sign = 0
            if dx > 0.25:
                dx_sign = 1
            elif dx < -0.25:
                dx_sign = -1

            if dx_sign != 0 and dx_sign == last_dx_sign:
                repeated_turn_frames += 1
            elif dx_sign != 0:
                repeated_turn_frames = 0
                last_dx_sign = dx_sign
            else:
                repeated_turn_frames = max(0, repeated_turn_frames - 1)

            # If the agent is not meaningfully moving, count low progress.
            meaningful_move = (
                "move_forward" in active
                or "strafe_left" in active
                or "strafe_right" in active
                or abs(dx) >= 0.35
            )

            if not meaningful_move:
                low_progress_frames += 1
            else:
                low_progress_frames = max(0, low_progress_frames - 1)

            # Start recovery when it freezes, spins/turns too long, or keeps trying blocked forward.
            repeated_noop = len(recent_active) >= 8 and all(a == "no_op" for a in list(recent_active)[-8:])
            repeated_forward_no_mouse = (
                len(recent_active) >= 10
                and sum(1 for a in list(recent_active)[-10:] if a in {"move_forward", "no_op"}) >= 9
                and abs(dx) < 0.35
            )

            if recovery_cooldown <= 0 and (
                repeated_noop
                or low_progress_frames >= LOW_PROGRESS_LIMIT
                or repeated_turn_frames >= TURN_LOCK_LIMIT
                or repeated_forward_no_mouse
            ):
                recovery_frames = ROUTE_RECOVERY_FRAMES
                recovery_cooldown = 24
                recovery_side *= -1
                low_progress_frames = 0
                repeated_turn_frames = 0
                recent_active.clear()

            # Recovery pattern:
            # unstick with short forward+strafe bursts, but DO NOT force mouse turn.
            # Forced mouse turn caused spinning in place.
            if recovery_frames > 0:
                active.discard("use")
                active.discard("mouse_left")
                active.discard("move_backward")

                active.add("move_forward")
                if recovery_side > 0:
                    active.discard("strafe_left")
                    active.add("strafe_right")
                else:
                    active.discard("strafe_right")
                    active.add("strafe_left")

                # Let the imitation model handle aiming/turning.
                # Recovery should not spin the mouse.
                dx = 0.0
                dy = 0.0
                recovery_frames -= 1

            # FORWARD_RECOVERY_FALLBACK
            # If backward is the top guess but blocked, avoid freezing forever.
            if not active and prob_by_name.get("move_forward", 0.0) >= 0.30:
                active.add("move_forward")

            if local_enabled:
                # Press/release keyboard buttons.
                for action, key in KEY_MAP.items():
                    if action in active and action not in prev_pressed:
                        kbd.press(key)
                    elif action not in active and action in prev_pressed:
                        kbd.release(key)

                # Mouse left click/hold.
                if "mouse_left" in active and "mouse_left" not in prev_pressed:
                    mse.press(mouse.Button.left)
                elif "mouse_left" not in active and "mouse_left" in prev_pressed:
                    mse.release(mouse.Button.left)

                # Mouse aim movement.
                if dx != 0.0 or dy != 0.0:
                    mse.move(int(dx), int(dy))

                prev_pressed = active
            else:
                release_all(kbd, mse)
                prev_pressed = set()

            if frame_idx % 20 == 0:
                top = sorted(prob_by_name.items(), key=lambda kv: kv[1], reverse=True)[:4]
                print(
                    f"[seq_agent] enabled={local_enabled} "
                    f"active={'+'.join(sorted(active)) if active else 'no_op'} "
                    f"mouse=({dx:.1f},{dy:.1f}) "
                    f"top={[(k, round(v, 2)) for k, v in top]}"
                )

            frame_idx += 1

            elapsed = time.time() - start
            time.sleep(max(0.0, 1 / 20 - elapsed))

    release_all(kbd, mse)
    listener.stop()
    print("[seq_agent] stopped cleanly")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[seq_agent] Ctrl+C received; exiting")
