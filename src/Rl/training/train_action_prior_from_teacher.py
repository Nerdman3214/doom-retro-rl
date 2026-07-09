import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imitation.teacher_rollout_dataset import TeacherRolloutDataset
from models.action_prior_model import ActionPriorCNN


ROLLOUT_ROOT = ROOT / "vision_dataset_v2" / "teacher_rollouts_balanced"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "action_prior_from_teacher_balanced.pt"


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TeacherRolloutDataset(
        rollout_root=ROLLOUT_ROOT,
        image_size=(84, 84),
    )

    num_actions = max(sample["action_index"] for sample in dataset.samples) + 1

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ActionPriorCNN(num_actions=num_actions).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs = 10

    print(f"[action_prior] samples={len(dataset)} train={train_size} val={val_size}")
    print(f"[action_prior] num_actions={num_actions}")
    print(f"[action_prior] device={device}")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, actions in train_loader:
            images = images.to(device)
            actions = actions.to(device)

            logits = model(images)
            loss = criterion(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == actions).sum().item())
            total += int(actions.numel())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for images, actions in val_loader:
                images = images.to(device)
                actions = actions.to(device)

                logits = model(images)
                loss = criterion(logits, actions)

                val_loss_total += float(loss.item()) * images.size(0)

                predictions = torch.argmax(logits, dim=1)
                val_correct += int((predictions == actions).sum().item())
                val_total += int(actions.numel())

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"[action_prior] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_actions": num_actions,
                    "image_size": (84, 84),
                    "best_val_acc": best_val_acc,
                },
                CHECKPOINT_PATH,
            )

            print(f"[action_prior] saved {CHECKPOINT_PATH}")

    print(f"[action_prior] best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
