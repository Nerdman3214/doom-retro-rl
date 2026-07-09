from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from models.action_prior_model import ActionPriorCNN


class ActionPriorAdvisor:
    """
    Loads the student action-prior model trained from ViZDoom teacher rollouts.

    This should be used as advice first, not as direct action override.
    """

    def __init__(self, checkpoint_path, action_names, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.action_names = list(action_names)
        self.enabled = False

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = None
        self.num_actions = len(self.action_names)

        self._load()

    def _load(self):
        if not self.checkpoint_path.exists():
            print(f"[action_prior] checkpoint not found: {self.checkpoint_path}")
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        num_actions = int(checkpoint.get("num_actions", self.num_actions))

        self.model = ActionPriorCNN(num_actions=num_actions).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.num_actions = num_actions
        self.enabled = True

        print(
            f"[action_prior] loaded {self.checkpoint_path} "
            f"num_actions={self.num_actions} device={self.device}"
        )

    def _preprocess(self, frame):
        if frame is None:
            return None

        image = frame

        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)

        # If frame is BGR from OpenCV/screenshot pipelines, this may not be perfect,
        # but it is okay for advice-only testing.
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()

        return image.to(self.device)

    def predict(self, frame):
        if not self.enabled or self.model is None:
            return {
                "enabled": False,
                "action_index": None,
                "action_name": None,
                "confidence": 0.0,
                "probs": {},
            }

        x = self._preprocess(frame)

        if x is None:
            return {
                "enabled": False,
                "action_index": None,
                "action_name": None,
                "confidence": 0.0,
                "probs": {},
            }

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        action_index = int(torch.argmax(probs).item())
        confidence = float(probs[action_index].item())

        if action_index < len(self.action_names):
            action_name = self.action_names[action_index]
        else:
            action_name = str(action_index)

        prob_dict = {}

        for i in range(min(len(self.action_names), probs.shape[0])):
            prob_dict[self.action_names[i]] = float(probs[i].item())

        return {
            "enabled": True,
            "action_index": action_index,
            "action_name": action_name,
            "confidence": confidence,
            "probs": prob_dict,
        }
