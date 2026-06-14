import torch
import torch.nn as nn


class ActionPriorCNN(nn.Module):
    """
    Small CNN that predicts the teacher's action from a frame.

    This is intentionally lightweight. It is not the PPO policy.
    It is a student imitation helper.
    """

    def __init__(self, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            feature_dim = self.features(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
