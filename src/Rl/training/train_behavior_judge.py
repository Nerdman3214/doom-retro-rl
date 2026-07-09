import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.behavior_judge_model import BehaviorJudgeCNN


DATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"
OUT_PATH = ROOT / "checkpoints" / "behavior_judge.pt"


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((84, 84))

    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return torch.tensor(arr, dtype=torch.float32)


def choose_three_frames(frames):
    """
    Pick early/mid/late frames so the model can see progress or stuck behavior.
    """
    n = len(frames)

    if n == 1:
        return [frames[0], frames[0], frames[0]]

    early_idx = max(0, int(n * 0.15))
    mid_idx = max(0, int(n * 0.50))
    late_idx = max(0, int(n * 0.85))

    return [frames[early_idx], frames[mid_idx], frames[late_idx]]


class SkillClipDataset(Dataset):
    def __init__(self, root: Path):
        self.items = []

        for label_path in sorted(root.glob("clip_*/labels.json")):
            clip_dir = label_path.parent
            frames_dir = clip_dir / "frames"

            with label_path.open("r", encoding="utf-8") as f:
                label = json.load(f)

            # Ignore original giant parent clips.
            num_frames = int(label.get("num_frames", 0))
            if num_frames > 300:
                continue

            frames = sorted(
                list(frames_dir.glob("*.png")) +
                list(frames_dir.glob("*.jpg")) +
                list(frames_dir.glob("*.jpeg"))
            )

            if not frames:
                continue

            quality = label.get("quality", "unknown")
            if quality not in {"good", "bad"}:
                continue

            y = 1 if quality == "good" else 0

            self.items.append({
                "clip_dir": clip_dir,
                "frames": frames,
                "label": y,
                "quality": quality,
                "skill": label.get("skill", "unknown"),
                "problem": label.get("problem", "none"),
                "source": label.get("source", "unknown"),
            })

        if not self.items:
            raise RuntimeError(f"No usable clips found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        selected = choose_three_frames(item["frames"])
        tensors = [preprocess_image(p) for p in selected]

        # [3, 3, 84, 84] -> [9, 84, 84]
        x = torch.cat(tensors, dim=0)
        y = torch.tensor(item["label"], dtype=torch.long)

        return x, y


def main():
    random.seed(42)
    torch.manual_seed(42)

    dataset = SkillClipDataset(DATA_ROOT)

    labels = [item["label"] for item in dataset.items]
    good = sum(1 for y in labels if y == 1)
    bad = sum(1 for y in labels if y == 0)

    print(f"[behavior_judge_3frame] usable_clips={len(dataset)} good={good} bad={bad}")

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[behavior_judge_3frame] device={device}")

    model = BehaviorJudgeCNN().to(device)

    total = good + bad
    bad_weight = total / max(1, 2 * bad)
    good_weight = total / max(1, 2 * good)

    class_weights = torch.tensor([bad_weight, good_weight], dtype=torch.float32).to(device)
    print(f"[behavior_judge_3frame] class_weights bad={bad_weight:.3f} good={good_weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            seen += x.size(0)

        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        model.eval()
        val_correct = 0
        val_seen = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)

                val_correct += (pred == y).sum().item()
                val_seen += x.size(0)

        val_acc = val_correct / max(1, val_seen)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": ["bad", "good"],
                    "input_size": [9, 84, 84],
                    "best_val_acc": best_val_acc,
                    "model_type": "behavior_judge_3frame",
                },
                OUT_PATH,
            )
            print(f"[behavior_judge_3frame] saved {OUT_PATH}")

    print(f"[behavior_judge_3frame] best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
