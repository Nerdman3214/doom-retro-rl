import torch
import torch.nn as nn


class BehaviorCloningModel(nn.Module):

    def __init__(self, action_space):

        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),    # normalise to 8×8 for any input size
        )


        self.fc = nn.Sequential(

            nn.Linear(64 * 8 * 8 + 2, 256),

            nn.ReLU(),

            nn.Linear(256, action_space)
        )


    def forward(self, frame, health, ammo):

        x = self.cnn(frame)

        x = x.view(x.size(0), -1)


        x = torch.cat(

            [x, health, ammo],

            dim=1

        )


        return self.fc(x)