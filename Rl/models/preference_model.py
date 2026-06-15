import torch
import torch.nn as nn


class PreferenceModel(nn.Module):

    def __init__(self):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(3, 16, 5, stride=2),

            nn.ReLU(),

            nn.Conv2d(16, 32, 5, stride=2),

            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2),

            nn.ReLU()
        )


        self.head = nn.Sequential(

            nn.Linear(64 * 8 * 8, 128),

            nn.ReLU(),

            nn.Linear(128, 1)
        )


    def forward(self, frame):

        x = self.encoder(frame)

        x = x.view(x.size(0), -1)

        return self.head(x)