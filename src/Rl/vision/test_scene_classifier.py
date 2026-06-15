from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "checkpoints" / "scene_classifier.pt"
RAW_DIR = ROOT_DIR / "vision_dataset" / "raw_frames"


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    images = sorted(
        list(RAW_DIR.glob("*.jpg"))
        + list(RAW_DIR.glob("*.jpeg"))
        + list(RAW_DIR.glob("*.png"))
        + list(RAW_DIR.glob("*.webp"))
    )

    if not images:
        print(f"No images found in {RAW_DIR}")
        return

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Classes: {classes}")
    print(f"Testing {min(50, len(images))} images...\n")

    for img_path in images[:50]:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = probs.max(dim=0)
        label = classes[top_idx.item()]

        top3 = torch.topk(probs, k=min(3, len(classes)))

        top3_text = []
        for prob, idx in zip(top3.values, top3.indices):
            top3_text.append(f"{classes[idx.item()]}={prob.item():.2f}")

        print(f"{img_path.name}: {label} ({top_prob.item():.2f}) | " + ", ".join(top3_text))


if __name__ == "__main__":
    main()