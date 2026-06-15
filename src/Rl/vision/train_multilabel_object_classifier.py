from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "vision_dataset_v2" / "object_multilabel"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "labels.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "object_multilabel_classifier.pt"
CLASS_MAP_PATH = CHECKPOINT_DIR / "object_multilabel_classes.json"

LABEL_COLUMNS = [
    "enemy_visible",
    "pickup_health",
    "pickup_ammo",
    "pickup_armor",
    "explosive_barrel",
    "no_important_object",
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42


class DoomObjectMultiLabelDataset(Dataset):
    def __init__(self, csv_path, image_dir, label_columns, transform=None):
        self.image_dir = Path(image_dir)
        self.label_columns = label_columns
        self.transform = transform
        self.rows = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.image_dir / row["filename"]
                if image_path.exists():
                    self.rows.append(row)

        if not self.rows:
            raise ValueError("No valid rows found for multi-label dataset.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")

        labels = torch.tensor(
            [float(row[col]) for col in self.label_columns],
            dtype=torch.float32,
        )

        if self.transform:
            image = self.transform(image)

        return image, labels


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


def build_model(num_labels):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def calculate_label_counts(csv_path):
    counts = {label: 0 for label in LABEL_COLUMNS}
    total = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            for label in LABEL_COLUMNS:
                counts[label] += int(row[label])

    return total, counts


def calculate_accuracy(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds == labels).float().mean().item()
    return correct


def main():
    seed_everything(SEED)

    total_rows, counts = calculate_label_counts(CSV_PATH)

    print("Multi-label dataset:")
    print(f"  rows: {total_rows}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(
            {
                "label_columns": LABEL_COLUMNS,
                "image_size": IMAGE_SIZE,
                "model_name": "resnet18",
                "task": "multi_label_object_detection",
            },
            f,
            indent=2,
        )

    train_tfms, val_tfms = build_transforms()

    full_dataset = DoomObjectMultiLabelDataset(
        CSV_PATH,
        IMAGE_DIR,
        LABEL_COLUMNS,
        transform=train_tfms,
    )

    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_dataset = DoomObjectMultiLabelDataset(
        CSV_PATH,
        IMAGE_DIR,
        LABEL_COLUMNS,
        transform=val_tfms,
    )
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

    model = build_model(num_labels=len(LABEL_COLUMNS)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(logits.detach(), labels)
            train_batches += 1

        train_loss /= train_batches
        train_acc /= train_batches

        model.eval()

        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_acc += calculate_accuracy(logits, labels)
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_label_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_label_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_columns": LABEL_COLUMNS,
                    "image_size": IMAGE_SIZE,
                    "model_name": "resnet18",
                    "task": "multi_label_object_detection",
                    "threshold": 0.5,
                },
                MODEL_PATH,
            )

            print(f"Saved best model to: {MODEL_PATH}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved class map to: {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()