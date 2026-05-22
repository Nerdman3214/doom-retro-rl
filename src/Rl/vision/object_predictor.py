from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = ROOT / "checkpoints" / "object_multilabel_classifier.pt"
DEFAULT_CLASS_PATH = ROOT / "checkpoints" / "object_multilabel_classes.json"


OBJECT_THRESHOLDS = {
    "enemy_visible": 0.50,
    "pickup_health": 0.35,
    "pickup_ammo": 0.25,
    "pickup_armor": 0.35,
    "explosive_barrel": 0.25,
    "no_important_object": 0.70,
}


class ObjectPredictor:
    """
    Multi-label object predictor for Doom/Freedoom frames.

    This model answers:
    - Is an enemy visible?
    - Is health visible?
    - Is ammo visible?
    - Is armor visible?
    - Is an explosive barrel visible?
    - Is no important object visible?

    It does NOT replace the scene/navigation classifier.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        class_path=DEFAULT_CLASS_PATH,
        device=None,
        thresholds=None,
    ):
        self.model_path = Path(model_path)
        self.class_path = Path(class_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.thresholds = thresholds or OBJECT_THRESHOLDS

        with open(self.class_path, "r") as f:
            class_data = json.load(f)

        self.label_columns = class_data["label_columns"]
        self.image_size = int(class_data.get("image_size", 224))

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.label_columns))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _to_pil(self, frame):
        """
        Accepts:
        - PIL Image
        - RGB numpy array
        - BGR numpy array from OpenCV

        Returns PIL RGB image.
        """

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if isinstance(frame, np.ndarray):
            if frame.ndim != 3:
                raise ValueError(f"Expected 3D image array, got shape={frame.shape}")

            # Heuristic:
            # Most env frames are RGB already.
            # If the caller passes OpenCV BGR, use predict_bgr().
            return Image.fromarray(frame.astype("uint8")).convert("RGB")

        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def predict(self, frame):
        image = self._to_pil(frame)
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu()

        scores = {
            label: float(probs[i].item())
            for i, label in enumerate(self.label_columns)
        }

        labels = {
            label: scores[label] >= self.thresholds.get(label, 0.50)
            for label in self.label_columns
        }

        return {
            "labels": labels,
            "scores": scores,
            "present": [label for label, value in labels.items() if value],
        }

    def predict_bgr(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.predict(frame_rgb)