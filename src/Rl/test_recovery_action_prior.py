from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import pandas as pd


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "recovery_action_prior.pt"
DATASET = ROOT / "vision_dataset" / "raw_human_play" / "vizdoom_wall_recovery_003"
CSV = DATASET / "actions.csv"


class SmallCNN(nn.Module):
    def __init__(self, num_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CKPT, map_location=device)
    action_names = ckpt["action_names"]

    model = SmallCNN(num_actions=len(action_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((84, 84)),
        T.ToTensor(),
    ])

    df = pd.read_csv(CSV)
    df = df[df["frame_path"].apply(lambda x: (DATASET / str(x)).exists())]

    print("[test_recovery_prior] loaded:", CKPT)
    print("[test_recovery_prior] rows with frames:", len(df))
    print("[test_recovery_prior] action_names:", action_names)
    print()

    sample = df.sample(min(20, len(df)), random_state=7)

    with torch.no_grad():
        for _, row in sample.iterrows():
            img_path = DATASET / row["frame_path"]
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

            pred = action_names[int(top_idx)]
            print(
                f"trap={row.get('trap', 'unknown'):<35} "
                f"mode={row.get('recovery_mode', 'unknown'):<15} "
                f"human={row['action']:<35} "
                f"pred={pred:<15} "
                f"conf={float(top_prob):.3f}"
            )


if __name__ == "__main__":
    main()
