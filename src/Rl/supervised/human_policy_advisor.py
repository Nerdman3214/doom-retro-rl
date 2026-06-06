from pathlib import Path

import cv2
import numpy as np
import torch

from supervised.train_human_policy import DoomHumanPolicyCNN
from supervised.human_policy_decoder import ACTION_NAMES, decode_teacher_buttons, load_thresholds


class HumanPolicyAdvisor:
    """
    Loads the supervised human policy and converts frames into safe teacher advice.

    This should be used as soft guidance:
      - small bonus if RL agrees
      - no hard override
    """

    def __init__(
        self,
        model_path="supervised_models/human_policy_cnn_best.pt",
        thresholds_path="supervised_models/button_thresholds.txt",
        device=None,
    ):
        self.root = Path(__file__).resolve().parents[1]
        self.model_path = self.root / model_path
        self.thresholds_path = self.root / thresholds_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.image_size = checkpoint.get("image_size", 84)

        self.model = DoomHumanPolicyCNN(action_dim=8).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.thresholds = load_thresholds(self.thresholds_path)

    def preprocess_frame(self, frame):
        if frame is None:
            return None

        frame = np.asarray(frame)

        # ViZDoom sometimes returns CHW.
        if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
            frame = np.transpose(frame[:3], (1, 2, 0))

        if frame.ndim != 3:
            return None

        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))

        x = torch.from_numpy(frame).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def advise_from_frame(self, frame):
        x = self.preprocess_frame(frame)
        if x is None:
            return {
                "buttons": [0] * 8,
                "label": "no_op",
                "confidence": 0.0,
                "probabilities": {},
            }

        logits = self.model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        buttons, label, prob_map = decode_teacher_buttons(
            probs,
            thresholds=self.thresholds,
            max_buttons=3,
        )

        confidence = 0.0
        active_probs = [
            prob_map[ACTION_NAMES[i]]
            for i, b in enumerate(buttons)
            if b
        ]

        if active_probs:
            confidence = float(max(active_probs))

        return {
            "buttons": buttons,
            "label": label,
            "confidence": confidence,
            "probabilities": prob_map,
        }

    def reward_for_action(self, rl_action_name, teacher_buttons, confidence=0.0):
        """
        Return small imitation/advice reward.

        This should stay small so it does not overpower:
          - exit progress
          - combat reward
          - survival
          - item/use reward
        """

        if rl_action_name not in ACTION_NAMES:
            return 0.0

        idx = ACTION_NAMES.index(rl_action_name)
        teacher_wants_action = bool(teacher_buttons[idx])

        if not teacher_wants_action:
            return 0.0

        # Confidence gate: avoid rewarding uncertain advice.
        if confidence < 0.55:
            return 0.0

        if rl_action_name == "shoot":
            return 0.05

        if rl_action_name == "use":
            return 0.08

        if rl_action_name == "move_forward":
            return 0.03

        if rl_action_name in {"turn_left", "turn_right"}:
            return 0.025

        if rl_action_name in {"strafe_left", "strafe_right", "move_backward"}:
            return 0.02

        return 0.0
