from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "checkpoints" / "object_classifier.pt"
CLASS_MAP_PATH = ROOT / "checkpoints" / "object_classifier_classes.json"
DATA_DIR = ROOT / "vision_dataset_v2" / "objects"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    with open(CLASS_MAP_PATH, "r") as f:
        class_data = json.load(f)

    idx_to_class = {
        int(idx): name
        for idx, name in class_data["idx_to_class"].items()
    }

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_class


def main():
    model, idx_to_class = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    all_images = []
    for path in DATA_DIR.glob("*/*.jpg"):
        all_images.append(path)

    sample_images = random.sample(all_images, min(25, len(all_images)))

    correct = 0

    for image_path in sample_images:
        true_label = image_path.parent.name

        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        pred_label = idx_to_class[pred_idx]

        if pred_label == true_label:
            correct += 1

        print(
            f"true={true_label:20s} "
            f"pred={pred_label:20s} "
            f"conf={confidence:.2f} "
            f"file={image_path.name}"
        )

    print(f"\nSample accuracy: {correct}/{len(sample_images)}")


if __name__ == "__main__":
    main()