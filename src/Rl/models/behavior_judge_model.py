import torch
import torch.nn as nn


class BehaviorJudgeCNN(nn.Module):
    """
    3-frame clip-level behavior judge.

    Input:
      [B, 9, 84, 84]

    9 channels = 3 RGB frames:
      early frame
      middle frame
      late frame

    Output:
      logits for 2 classes:
        0 = bad
        1 = good
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 9, 84, 84)
            feature_dim = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))
