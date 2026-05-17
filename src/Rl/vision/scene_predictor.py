from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ScenePredictor:
    """
    Runtime vision model for DoomEnv.

    It converts a frame into a scene label such as:
    enemy, front_wall, open_path, door_or_button, damage_or_death, unclear.

    This should be used as perception support, not as the whole AI.
    DoomEnv/PPO should still decide what action to take.
    """

    def __init__(self, model_path=None, device=None):
        root_dir = Path(__file__).resolve().parents[1]

        if model_path is None:
            model_path = root_dir / "checkpoints" / "scene_classifier.pt"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing scene classifier: {self.model_path}")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.classes = checkpoint["classes"]
        self.image_size = checkpoint.get("image_size", 224)

        self.model = self._build_model(len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        print(f"[vision] Loaded scene classifier: {self.model_path}")
        print(f"[vision] Classes: {self.classes}")

    def _build_model(self, num_classes):
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    def predict(self, frame):
        """
        Predict the scene class from a BGR OpenCV frame.
        Returns:
            {
                "label": str,
                "confidence": float,
                "probs": dict
            }
        """
        if frame is None or frame.size == 0:
            return {
                "label": "unclear",
                "confidence": 0.0,
                "probs": {},
            }

        # OpenCV uses BGR. PIL expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs_tensor = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = probs_tensor.max(dim=0)

        label = self.classes[top_idx.item()]
        confidence = float(top_prob.item())

        probs = {
            class_name: float(probs_tensor[i].item())
            for i, class_name in enumerate(self.classes)
        }

        return {
            "label": label,
            "confidence": confidence,
            "probs": probs,
        }