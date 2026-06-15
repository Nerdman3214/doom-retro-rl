from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "checkpoints" / "object_multilabel_classifier.pt"
CLASS_MAP_PATH = ROOT / "checkpoints" / "object_multilabel_classes.json"

DATA_DIR = ROOT / "vision_dataset_v2" / "object_multilabel"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "labels.csv"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    with open(CLASS_MAP_PATH, "r") as f:
        class_data = json.load(f)

    label_columns = class_data["label_columns"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_columns


def load_rows():
    rows = []

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = IMAGE_DIR / row["filename"]
            if image_path.exists():
                rows.append(row)

    return rows


def main():
    model, label_columns = load_model()
    rows = load_rows()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    sample_rows = random.sample(rows, min(20, len(rows)))

    for row in sample_rows:
        image_path = IMAGE_DIR / row["filename"]

        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0]

        true_labels = [
            label
            for label in label_columns
            if int(row[label]) == 1
        ]

        predicted_labels = []
        scored_labels = []

        for i, label in enumerate(label_columns):
            score = float(probs[i].item())
            scored_labels.append((label, score))

            if score >= 0.50:
                predicted_labels.append(label)

        scored_labels.sort(key=lambda item: item[1], reverse=True)

        print("=" * 80)
        print(f"file: {row['filename']}")
        print(f"true: {true_labels}")
        print(f"pred: {predicted_labels}")
        print("scores:")
        for label, score in scored_labels:
            print(f"  {label:20s} {score:.3f}")


if __name__ == "__main__":
    main()