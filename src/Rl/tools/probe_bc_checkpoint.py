#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.train_vizdoom_video_behavior_clone import VideoBCNet


def image_to_tensor(path, image_size, scale_mode):
    img = Image.open(path).convert("RGB")

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    img = img.resize((image_size, image_size), resample)
    arr = np.asarray(img, dtype=np.float32)

    if scale_mode == "zero_one":
        arr = arr / 255.0
    elif scale_mode == "minus_one_one":
        arr = (arr / 127.5) - 1.0
    elif scale_mode == "raw_255":
        pass
    else:
        raise ValueError(scale_mode)

    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def row_action(row, action_names):
    if "action" in row and row["action"]:
        return row["action"]

    active = []
    for name in action_names:
        if row.get(name, "0") in ("1", "true", "True", "yes"):
            active.append(name)
    return "+".join(active) if active else "no_op"


def run_probe(args, scale_mode):
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    seq_len = int(ckpt.get("seq_len", 8))
    image_size = int(ckpt.get("image_size", 84))
    action_names = ckpt["action_names"]

    model = VideoBCNet(num_actions=len(action_names))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    session = Path(args.session)
    actions_csv = session / "actions.csv"
    frames_dir = session / "frames"
    rows = list(csv.DictReader(actions_csv.open()))

    print()
    print("====================================")
    print("[probe] scale_mode:", scale_mode)
    print("[probe] checkpoint:", args.checkpoint)
    print("[probe] session:", session)
    print("[probe] rows:", len(rows))
    print("[probe] seq_len:", seq_len)
    print("[probe] image_size:", image_size)
    print("[probe] action_names:", action_names)

    indexes = [
        seq_len,
        25,
        75,
        174,
        349,
        524,
        len(rows) - 1,
    ]

    for idx in indexes:
        idx = max(seq_len, min(idx, len(rows) - 1))
        frame_tensors = []

        for j in range(idx - seq_len + 1, idx + 1):
            p = frames_dir / f"frame_{j:06d}.png"

            if not p.exists():
                print("[probe] missing frame:", p)
                return

            frame_tensors.append(image_to_tensor(p, image_size, scale_mode))

        x = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(args.device)

        with torch.no_grad():
            out = model(x)
            logits = out[0].detach().cpu()
            probs = torch.sigmoid(logits)

        pred = [name for name, prob in zip(action_names, probs) if prob >= args.threshold]

        print()
        print(f"[probe] frame={idx}")
        print("[probe] true action:", row_action(rows[idx], action_names))
        print("[probe] predicted:", "+".join(pred) if pred else "no_op")
        print("[probe] logits:", ", ".join(f"{name}={float(v):+.4f}" for name, v in zip(action_names, logits)))
        print("[probe] probs :", ", ".join(f"{name}={float(v):.6f}" for name, v in zip(action_names, probs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vizdoom_video_behavior_clone_success_only.pt")
    ap.add_argument("--session", default="vision_dataset/raw_human_play/session_z_20260619_161751")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    for scale_mode in ["zero_one", "minus_one_one"]:
        run_probe(args, scale_mode)


if __name__ == "__main__":
    main()
