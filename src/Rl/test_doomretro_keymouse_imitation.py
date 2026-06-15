from pathlib import Path
import random

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "doomretro_keymouse_imitation_cnn.pt"
DATA_ROOT = ROOT / "vision_dataset_v2" / "doomretro_keymouse_imitation"

BUTTON_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "use",
    "mouse_left",
]


class KeyMouseCNN(nn.Module):
    def __init__(self, num_buttons):
        super().__init__()
        self.backbone = nn.Sequential(
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
        )
        self.button_head = nn.Linear(512, num_buttons)
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.button_head(z), self.mouse_head(z)


def keyboard_action_to_buttons(action, mouse_left):
    parts = set(str(action).split("+"))
    buttons = []

    for name in BUTTON_NAMES:
        if name == "mouse_left":
            if int(mouse_left) > 0:
                buttons.append(name)
        elif name in parts:
            buttons.append(name)

    return "+".join(buttons) if buttons else "no_op"


def pred_to_action(button_probs, mouse_pred, threshold=0.5):
    active = []

    for i, name in enumerate(BUTTON_NAMES):
        if float(button_probs[i]) >= threshold:
            active.append(name)

    mouse_dx = float(mouse_pred[0]) * 80.0
    mouse_dy = float(mouse_pred[1]) * 80.0

    return "+".join(active) if active else "no_op", mouse_dx, mouse_dy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CKPT, map_location=device)

    button_names = ckpt.get("button_names", BUTTON_NAMES)

    model = KeyMouseCNN(num_buttons=len(button_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((84, 84)),
        T.ToTensor(),
    ])

    rows = []

    for run in sorted(DATA_ROOT.glob("doomretro_easiest_keymouse_completion_*")):
        csv_path = run / "actions.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 300:
            continue

        for _, row in df.iterrows():
            frame_path = run / str(row["frame_path"])
            if not frame_path.exists():
                continue

            rows.append({
                "run": run.name,
                "frame_path": frame_path,
                "keyboard_action": row.get("keyboard_action", "no_op"),
                "mouse_left": int(row.get("mouse_left", 0)),
                "mouse_dx": float(row.get("mouse_dx", 0.0)),
                "mouse_dy": float(row.get("mouse_dy", 0.0)),
            })

    print("[keymouse_test] checkpoint:", CKPT)
    print("[keymouse_test] usable samples:", len(rows))
    print("[keymouse_test] button_names:", button_names)
    print()

    sample = random.sample(rows, min(50, len(rows)))

    exact = 0

    with torch.no_grad():
        for item in sample:
            img = Image.open(item["frame_path"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            button_logits, mouse_pred = model(x)
            probs = torch.sigmoid(button_logits)[0].detach().cpu()
            mouse = mouse_pred[0].detach().cpu()

            pred_action, pred_dx, pred_dy = pred_to_action(probs, mouse)
            human_action = keyboard_action_to_buttons(item["keyboard_action"], item["mouse_left"])

            if pred_action == human_action:
                exact += 1

            print(
                f"run={item['run']:<42} "
                f"human={human_action:<35} "
                f"pred={pred_action:<35} "
                f"human_mouse=({item['mouse_dx']:>6.1f},{item['mouse_dy']:>6.1f}) "
                f"pred_mouse=({pred_dx:>6.1f},{pred_dy:>6.1f})"
            )

    print()
    print(f"[keymouse_test] exact={exact}/{len(sample)} = {exact / max(len(sample), 1):.3f}")


if __name__ == "__main__":
    main()
