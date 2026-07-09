import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import torch
import torch.optim as optim

from models.preference_model import PreferenceModel
from datasets.trajectory_dataset import TrajectoryDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = TrajectoryDataset()

model = PreferenceModel().to(DEVICE)

optimizer = optim.Adam(

    model.parameters(),

    lr=0.0001
)


for step in range(2000):

    traj_a, traj_b, label = dataset.sample_pair()


    frame_a = torch.tensor(

        traj_a["frames"][0]

    ).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


    frame_b = torch.tensor(

        traj_b["frames"][0]

    ).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


    score_a = model(frame_a)

    score_b = model(frame_b)


    loss = torch.log(

        1 + torch.exp(

            -(score_a - score_b)

        )

    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 100 == 0:

        print("Step:", step, "Loss:", loss.item())


torch.save(

    model.state_dict(),

    "preference_model.pt"
)