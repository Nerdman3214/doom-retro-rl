import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "vision_dataset" / "classified"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "scene_classifier.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42


def count_images_by_class(dataset):
    counts = {class_name: 0 for class_name in dataset.classes}

    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        counts[class_name] += 1

    return counts


def build_model(num_classes):
    """
    Build a lightweight transfer-learning classifier.

    MobileNetV3 is small enough for quick training but strong enough to learn
    simple scene classes like open_path, wall, enemy_close, and door_or_button.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {DATA_DIR}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

    if len(dataset) < 20:
        print("\nNot enough labeled images yet.")
        print(f"Found only {len(dataset)} labeled images.")
        print("Label at least 20 images first just to test training.")
        print("For a useful first model, aim for 300-500 labeled images.\n")
        return

    print("\nClasses:")
    for idx, class_name in enumerate(dataset.classes):
        print(f"  {idx}: {class_name}")

    print("\nImage counts:")
    counts = count_images_by_class(dataset)
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")

    num_classes = len(dataset.classes)

    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": dataset.classes,
                "image_size": IMAGE_SIZE,
                "best_val_acc": best_val_acc,
            }

            torch.save(checkpoint, MODEL_PATH)

            print(f"Saved best model to: {MODEL_PATH}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()