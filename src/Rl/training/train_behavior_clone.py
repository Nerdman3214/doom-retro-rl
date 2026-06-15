import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.behavior_cloning_model import BehaviorCloningModel
from action.action_space import ActionSpace

ACTION_TO_IDX = {a: i for i, a in enumerate(ActionSpace.ACTIONS)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_WEIGHTS = {
    "enemy_seen":        5.0,  # Boosted for stronger shoot confidence
    "took_damage":       3.0,
    "low_health_escape": 4.0,
    "door_open":         2.0,
    "pickup_collected":  2.0,
    "none":              1.0,
}

# Action-specific multipliers to boost confidence for critical actions
ACTION_WEIGHTS = {
    "shoot": 2.0,       # Boost shoot action
    "use": 2.5,         # Boost use action more aggressively
}

OUT_W, OUT_H = 84, 84


class EventDemoDataset(Dataset):
    """Loads a human demo pkl.

    Accepts two formats:
      - event-tagged: list of (frame, key_list, event_str)
      - plain recording: list of (frame, key_list)
    """

    def __init__(self, pkl_path):
        # Support both a single list dump and a streaming dump (one pickle
        # object per frame, as written by record_human_session.py).
        # The recorder uses a persistent Pickler so we must use a persistent
        # Unpickler to preserve memo state across frames.
        raw = []
        with open(pkl_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            try:
                first = unpickler.load()
            except EOFError:
                first = []
            if isinstance(first, list):
                # Old format: entire dataset saved as one list
                raw = first
            else:
                # Streaming format: each frame is a separate pickled object
                raw.append(first)
                try:
                    while True:
                        raw.append(unpickler.load())
                except EOFError:
                    pass

        self.samples = []
        for entry in raw:
            if isinstance(entry, tuple) and len(entry) == 3:
                frame, keys, event = entry
            elif isinstance(entry, tuple) and len(entry) == 2:
                frame, keys = entry
                event = "none"
            else:
                continue  # skip malformed entries

            action_name = self._keys_to_action(keys)
            if action_name is None:
                continue
            weight = EVENT_WEIGHTS.get(event, 1.0)
            # Apply action-specific boosting for critical actions
            if action_name in ACTION_WEIGHTS:
                weight *= ACTION_WEIGHTS[action_name]
            self.samples.append((frame, action_name, weight))

        print(f"Loaded {len(self.samples)} samples from {pkl_path}.")

    def _keys_to_action(self, keys):
        """Map raw pynput key strings to an action name."""
        shoot = "Key.ctrl_l" in keys or "Key.ctrl_r" in keys
        if "Key.up" in keys or "'w'" in keys:
            return "move_forward_shoot" if shoot else "move_forward"
        if "Key.down" in keys or "'s'" in keys:
            return "move_backward_shoot" if shoot else "move_backward"
        if "Key.left" in keys or "'a'" in keys:
            return "turn_left_shoot" if shoot else "turn_left"
        if "Key.right" in keys or "'d'" in keys:
            return "turn_right_shoot" if shoot else "turn_right"
        if shoot:
            return "shoot"
        if "Key.space" in keys or "Key.e" in keys or "'e'" in keys:
            return "use"
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, action_name, weight = self.samples[idx]
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))   # HWC -> CHW
        return (
            torch.tensor(frame, dtype=torch.float32),
            torch.tensor(ACTION_TO_IDX[action_name], dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )


def train():
    ACTION_SPACE = len(ActionSpace.ACTIONS)
    BATCH_SIZE = 32
    EPOCHS = 15  # Increased epochs for better convergence with boosted weights

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Prefer event-tagged demo; fall back to plain human recording.
    for filename in ("human_event_demo.pkl", "human_demo.pkl"):
        pkl_path = os.path.join(base_dir, filename)
        if os.path.exists(pkl_path):
            print(f"Using demo file: {pkl_path}")
            break
    else:
        print("ERROR: No demo file found. Looking for:")
        print(f"  {os.path.join(base_dir, 'human_event_demo.pkl')}")
        print(f"  {os.path.join(base_dir, 'human_demo.pkl')}")
        print("Record yourself playing first:")
        print("  python recording/record_human_session.py")
        return

    dataset = EventDemoDataset(pkl_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BehaviorCloningModel(ACTION_SPACE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    base_criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(EPOCHS):
        total_loss = 0

        for frame, action, weight in dataloader:
            frame  = frame.to(DEVICE, non_blocking=True)
            action = action.to(DEVICE, non_blocking=True)
            weight = weight.to(DEVICE, non_blocking=True)

            # BC model expects (frame, health, ammo) — pass dummy scalars
            health = torch.ones(frame.size(0), 1, device=DEVICE)
            ammo   = torch.ones(frame.size(0), 1, device=DEVICE) * 0.5

            logits = model(frame, health, ammo)
            loss   = (base_criterion(logits, action) * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", "behavior_clone_model.pt",
    )
    torch.save(model.state_dict(), out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    train()