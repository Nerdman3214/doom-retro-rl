import sys
import os
import glob
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preference_model import PreferenceModel


TRAJ_DIR = "trajectories"
PREFS_FILE = "human_preferences.pkl"
CLIP_LEN = 16
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.0003


def load_trajectories():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "traj_*.npz")))
    trajs = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        frames = data["frames"]
        rewards = data["rewards"]
        if len(frames) >= CLIP_LEN:
            trajs.append({"frames": frames, "rewards": rewards})
    return trajs


def sample_clip(traj):
    start = random.randint(0, len(traj["frames"]) - CLIP_LEN)
    frames = traj["frames"][start:start + CLIP_LEN]
    rewards = traj["rewards"][start:start + CLIP_LEN]
    return frames, float(np.sum(rewards))


def load_human_preferences():
    """Load human preference pairs from human_preferences.pkl.

    Expected format: list of dicts with keys:
        - "clip1_frames": np.array of frames (N, H, W, C)
        - "clip2_frames": np.array of frames (N, H, W, C)
        - "label": float (1.0 = clip1 preferred, 0.0 = clip2 preferred)
    """
    with open(PREFS_FILE, "rb") as f:
        prefs = pickle.load(f)
    pairs = []
    for p in prefs:
        pairs.append((p["clip1_frames"], p["clip2_frames"], p["label"]))
    return pairs


def build_pairs_auto(trajs, n_pairs=500):
    """Fallback: build preference pairs using cumulative reward."""
    pairs = []
    for _ in range(n_pairs):
        t1 = random.choice(trajs)
        t2 = random.choice(trajs)
        clip1, ret1 = sample_clip(t1)
        clip2, ret2 = sample_clip(t2)
        label = 1.0 if ret1 >= ret2 else 0.0
        pairs.append((clip1, clip2, label))
    return pairs


class PreferencePairDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clip1, clip2, label = self.pairs[idx]
        # Use middle frame as representative
        mid = len(clip1) // 2
        f1 = torch.tensor(clip1[mid], dtype=torch.float32).permute(2, 0, 1)
        f2 = torch.tensor(clip2[mid], dtype=torch.float32).permute(2, 0, 1)
        return f1, f2, torch.tensor(label, dtype=torch.float32)


def train():
    if os.path.exists(PREFS_FILE):
        pairs = load_human_preferences()
        print(f"Loaded {len(pairs)} human preference pairs from {PREFS_FILE}.")
    else:
        trajs = load_trajectories()
        if len(trajs) < 2:
            print(f"Need at least 2 trajectories in {TRAJ_DIR}/, "
                  f"found {len(trajs)}.")
            print("Run train_rl_agent.py first to collect data.")
            return
        print(f"Loaded {len(trajs)} trajectories.")
        pairs = build_pairs_auto(trajs)
        print(f"Built {len(pairs)} pairs (auto-labeled, no "
              f"{PREFS_FILE} found).")

    dataset = PreferencePairDataset(pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PreferenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for f1, f2, labels in loader:
            f1, f2, labels = f1.to(device), f2.to(device), labels.to(device)

            s1 = model(f1).squeeze(-1)
            s2 = model(f2).squeeze(-1)

            # Bradley-Terry: P(clip1 preferred) = sigmoid(s1 - s2)
            logits = s1 - s2
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss={total_loss / total:.4f}  acc={acc:.3f}")

    torch.save(model.state_dict(), "checkpoints/preference_model.pt")
    print("Saved checkpoints/preference_model.pt")


if __name__ == "__main__":
    train()
