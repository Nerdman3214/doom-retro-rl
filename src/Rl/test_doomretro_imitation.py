from pathlib import Path
import random

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "doomretro_imitation_cnn.pt"
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keyed_imitation"


class ImitationCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def simplify_action(action: str):
    parts = set(str(action).split("+"))

    if "use" in parts:
        return "use"
    if "shoot" in parts:
        return "shoot"
    if "move_forward" in parts:
        return "move_forward"
    if "strafe_left" in parts:
        return "strafe_left"
    if "strafe_right" in parts:
        return "strafe_right"
    if "turn_left" in parts:
        return "turn_left"
    if "turn_right" in parts:
        return "turn_right"
    if "move_backward" in parts:
        return "move_backward"

    return "no_op"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CKPT, map_location=device)
    action_names = ckpt["action_names"]

    model = ImitationCNN(num_actions=len(action_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((84, 84)),
        T.ToTensor(),
    ])

    rows = []

    for run in sorted(DATA_ROOT.glob("doomretro_easiest_level_completion_*")):
        csv_path = run / "actions.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            frame_path = run / str(row["frame_path"])
            if not frame_path.exists():
                continue

            rows.append({
                "run": run.name,
                "frame_path": frame_path,
                "human_raw": row["action"],
                "human_simple": simplify_action(row["action"]),
            })

    print("[test_imitation] checkpoint:", CKPT)
    print("[test_imitation] samples:", len(rows))
    print("[test_imitation] action_names:", action_names)
    print()

    sample = random.sample(rows, min(40, len(rows)))

    correct = 0

    with torch.no_grad():
        for item in sample:
            img = Image.open(item["frame_path"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

            pred = action_names[int(top_idx)]
            human = item["human_simple"]

            if pred == human:
                correct += 1

            print(
                f"run={item['run']:<40} "
                f"human={human:<15} "
                f"raw={item['human_raw']:<30} "
                f"pred={pred:<15} "
                f"conf={float(top_prob):.3f}"
            )

    print()
    print(f"[test_imitation] sample_acc={correct}/{len(sample)} = {correct / max(len(sample), 1):.3f}")


if __name__ == "__main__":
    main()
