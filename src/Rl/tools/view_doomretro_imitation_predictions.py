from pathlib import Path
import argparse
import time

import cv2
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


ROOT = Path(__file__).resolve().parents[1]
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


def draw_text(img, lines):
    y = 28
    for line in lines:
        cv2.putText(
            img,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="doomretro_easiest_level_completion_002")
    parser.add_argument("--delay", type=float, default=0.04)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    run_dir = DATA_ROOT / args.run
    csv_path = run_dir / "actions.csv"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    if not CKPT.exists():
        raise SystemExit(f"Missing checkpoint: {CKPT}")

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

    df = pd.read_csv(csv_path)

    print("[viewer] run:", args.run)
    print("[viewer] rows:", len(df))
    print("[viewer] checkpoint:", CKPT)
    print("[viewer] controls: q/ESC = quit, space = pause/resume")
    print()

    paused = False

    for i in range(args.start, len(df), max(args.step, 1)):
        row = df.iloc[i]
        frame_path = run_dir / str(row["frame_path"])

        if not frame_path.exists():
            continue

        pil_img = Image.open(frame_path).convert("RGB")
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        pred = action_names[int(top_idx)]
        conf = float(top_prob.detach().cpu().item())

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Resize for easier viewing.
        h, w = frame.shape[:2]
        max_w = 1280
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        human_raw = str(row["action"])
        human_simple = simplify_action(human_raw)

        match = "MATCH" if pred == human_simple else "DIFF"

        lines = [
            f"frame={i} run={args.run}",
            f"human raw: {human_raw}",
            f"human simplified: {human_simple}",
            f"agent pred: {pred} conf={conf:.3f} {match}",
        ]

        frame = draw_text(frame, lines)

        cv2.imshow("Doom Retro Imitation Prediction Viewer", frame)

        while True:
            key = cv2.waitKey(1 if paused else int(args.delay * 1000)) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

            if key == ord(" "):
                paused = not paused
                continue

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
