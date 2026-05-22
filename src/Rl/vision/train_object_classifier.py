from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "vision_dataset_v2" / "objects"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "object_classifier.pt"
CLASS_MAP_PATH = CHECKPOINT_DIR / "object_classifier_classes.json"


IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.25),
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
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, val_tfms


def count_images_by_class(data_dir):
    counts = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            counts[class_dir.name] = count
    return counts


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def main():
    seed_everything(SEED)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    counts = count_images_by_class(DATA_DIR)
    print("Object dataset counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    empty_classes = [name for name, count in counts.items() if count == 0]
    if empty_classes:
        raise ValueError(f"Empty class folders found. Move these out first: {empty_classes}")

    train_tfms, val_tfms = build_transforms()

    full_dataset_for_classes = datasets.ImageFolder(DATA_DIR)
    class_to_idx = full_dataset_for_classes.class_to_idx
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            },
            f,
            indent=2,
        )

    print("\nClasses:")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfms)

    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Validation dataset needs validation transforms.
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfms)
    val_subset.dataset = val_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = build_model(num_classes=len(class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": IMAGE_SIZE,
                    "model_name": "resnet18",
                },
                MODEL_PATH,
            )

            print(f"Saved best model to: {MODEL_PATH}")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Class map saved to: {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()