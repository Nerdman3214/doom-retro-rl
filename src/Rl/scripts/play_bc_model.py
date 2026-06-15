"""
Run the behavior-clone model to play DOOM Retro.

The BC model was trained on real human gameplay so it should produce
actual movement — unlike the degenerate PPO checkpoint.

Usage:
    1. Start DOOM Retro manually and load a map.
    2. python scripts/play_bc_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import frame_cache
import frame_processor
from controller.doom_controller import DoomController
from models.behavior_cloning_model import BehaviorCloningModel
from action.action_space import ActionSpace

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "behavior_clone_model.pt",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS = ActionSpace.ACTIONS
OUT_W, OUT_H = 84, 84


def get_frame():
    raw, w, h = frame_cache.get_raw()
    resized = frame_processor.resize_bgra_to_bgr(raw, w, h, OUT_W, OUT_H)
    return np.frombuffer(resized, dtype=np.uint8).reshape(OUT_H, OUT_W, 3).copy()


def frame_to_tensor(frame):
    """HWC uint8 → 1×C×H×W float32 in [0,1]."""
    t = torch.from_numpy(frame.astype(np.float32) / 255.0)   # H×W×3
    t = t.permute(2, 0, 1).unsqueeze(0)                       # 1×3×H×W
    return t.to(DEVICE)


def main():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Run training/train_behavior_clone.py first.")
        return

    print(f"Loading BC model from {CHECKPOINT} on {DEVICE} ...")
    model = BehaviorCloningModel(len(ACTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    controller = DoomController()
    if not controller.window_id:
        print("ERROR: DOOM Retro window not found. Start the game first.")
        return

    # Placeholder health/ammo tensors (we don't have HUD OCR yet)
    health = torch.tensor([[1.0]], dtype=torch.float32).to(DEVICE)   # 100/100
    ammo   = torch.tensor([[0.5]], dtype=torch.float32).to(DEVICE)   # 50/100

    print("BC agent playing DOOM. Press Ctrl+C to stop.\n")
    action_counts = {a: 0 for a in ACTIONS}

    try:
        step = 0
        while True:
            frame_cache.invalidate()
            frame = get_frame()
            t_frame = frame_to_tensor(frame)

            with torch.no_grad():
                logits = model(t_frame, health, ammo)
                action_idx = logits.argmax(dim=1).item()

            action_name = ACTIONS[action_idx]
            action_counts[action_name] += 1
            step += 1

            # Print action distribution every 50 steps
            if step % 50 == 0:
                total = sum(action_counts.values())
                print(f"Step {step} | last: {action_name}")
                for a, c in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
                    print(f"  {a:25s} {c/total*100:.1f}%")

            # Execute the action via the controller.
            action_map = {
                "move_forward": "start_move_forward",
                "move_backward": "start_move_backward",
                "turn_left": "start_turn_left",
                "turn_right": "start_turn_right",
                "strafe_left": "start_strafe_left",
                "strafe_right": "start_strafe_right",
                "shoot": "shoot",
                "use": "use",
                "swap_weapon": "swap_weapon",
                "melee_attack": None,
            }

            method_name = action_map.get(action_name)
            if method_name is not None and hasattr(controller, method_name):
                getattr(controller, method_name)()
            elif action_name == "melee_attack":
                # Controller does not support melee attack, so do nothing.
                pass
            else:
                print(f"[play_bc_model] Unsupported action: {action_name}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
