import torch
import torch.nn as nn
import os



class PolicyNetwork(nn.Module):

    def __init__(self, action_space):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 8 * 8 + 2, 256),
            nn.ReLU(),

            nn.Linear(256, action_space)
        )

    def forward(self, frame, health, ammo):

        x = self.cnn(frame)

        x = x.view(x.size(0), -1)

        x = torch.cat([x, health, ammo], dim=1)

        return self.fc(x)

    def load_behavior_clone_weights(self, path):

        self.load_state_dict(
            torch.load(path, map_location="cpu"),
            strict=False
        )

        print("Behavior cloning weights loaded successfully.")
