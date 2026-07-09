from pathlib import Path
from collections import Counter
import argparse
import csv
import random
import json

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from training.train_vizdoom_video_behavior_clone import VideoBCNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]

ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "shoot",
    "use",
]

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def parse_buttons(text):
    text = str(text or "no_op").strip()
    parts = set()
    for chunk in text.replace(",", "+").split("+"):
        chunk = chunk.strip()
        if chunk:
            parts.add(chunk)

    y = torch.zeros(len(ACTION_NAMES), dtype=torch.float32)

    for p in parts:
        if p in ACTION_TO_IDX:
            y[ACTION_TO_IDX[p]] = 1.0
        elif p in ("mouse_left", "left", "fire"):
            y[ACTION_TO_IDX["shoot"]] = 1.0

    return y, parts


def load_rows(csv_path):
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def frame_to_tensor(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class StrictSuccessVideoDataset(Dataset):
    def __init__(self, data_root, seq_len=8, image_size=84, idle_keep=0.20):
        self.data_root = Path(data_root)
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.samples = []

        runs = sorted(self.data_root.glob("session_ai_success_*"))
        if not runs:
            runs = sorted(self.data_root.glob("*"))

        action_counts = Counter()
        kept_by_run = Counter()
        skipped = []

        for run in runs:
            csv_path = run / "actions.csv"
            if not csv_path.exists():
                continue

            rows = load_rows(csv_path)
            usable = []

            for row in rows:
                frame_rel = row.get("frame_path", "")
                frame_path = run / frame_rel
                if not frame_path.exists():
                    continue

                label_text = row.get("buttons") or row.get("action") or "no_op"
                y, parts = parse_buttons(label_text)

                is_idle = float(y.sum().item()) == 0.0
                if is_idle and random.random() > idle_keep:
                    continue

                try:
                    wp_idx = int(float(row.get("wp_idx", -1)))
                except Exception:
                    wp_idx = -1

                # Bias the learner toward important terminal/door/combat behavior.
                weight = 1.0
                if "use" in parts:
                    weight = max(weight, 3.0)
                if "shoot" in parts:
                    weight = max(weight, 1.5)
                if wp_idx >= 240:
                    weight = max(weight, 3.0)

                usable.append({
                    "frame_path": frame_path,
                    "target": y,
                    "weight": float(weight),
                    "run": run.name,
                    "wp_idx": wp_idx,
                    "action": label_text,
                })

                for name, idx in ACTION_TO_IDX.items():
                    if y[idx].item() > 0:
                        action_counts[name] += 1

            if len(usable) < self.seq_len:
                skipped.append((run.name, len(usable)))
                continue

            for i in range(self.seq_len - 1, len(usable)):
                seq = usable[i - self.seq_len + 1:i + 1]
                target = usable[i]
                self.samples.append({
                    "seq_paths": [s["frame_path"] for s in seq],
                    "target": target["target"],
                    "weight": target["weight"],
                    "run": target["run"],
                    "wp_idx": target["wp_idx"],
                    "action": target["action"],
                })
                kept_by_run[run.name] += 1

        if not self.samples:
            raise SystemExit(f"No usable samples found in {self.data_root}")

        print(f"[strict_video_bc] runs={len(kept_by_run)} samples={len(self.samples)}")
        print("[strict_video_bc] action_counts:", dict(action_counts))
        print("[strict_video_bc] skipped_runs:", skipped[:20])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = [frame_to_tensor(p, self.image_size) for p in s["seq_paths"]]
        x = torch.stack(frames, dim=0)
        y = s["target"]
        w = torch.tensor(float(s["weight"]), dtype=torch.float32)
        return x, y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="vision_dataset/success_ai_play_strict_clean_h40_v1")
    ap.add_argument("--out", default="checkpoints/vizdoom_video_behavior_clone_strict_success_h40_v1.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=84)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = ROOT / args.data_root
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = StrictSuccessVideoDataset(
        data_root=data_root,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )

    val_size = max(1, int(len(ds) * args.val_frac))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoBCNet(num_actions=len(ACTION_NAMES)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0

        for x, y, w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device).view(-1, 1)

            logits = model(x)
            loss_per_action = bce(logits, y)
            loss = (loss_per_action * w).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            train_seen += x.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0

        with torch.no_grad():
            for x, y, w in val_loader:
                x = x.to(device)
                y = y.to(device)
                w = w.to(device).view(-1, 1)

                logits = model(x)
                loss = (bce(logits, y) * w).mean()

                val_loss_sum += float(loss.item()) * x.size(0)
                val_seen += x.size(0)

        train_loss = train_loss_sum / max(1, train_seen)
        val_loss = val_loss_sum / max(1, val_seen)

        print(
            f"[strict_video_bc] epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "action_names": ACTION_NAMES,
                "seq_len": args.seq_len,
                "image_size": args.image_size,
                "data_root": str(data_root),
                "best_val_loss": best_val,
                "train_size": train_size,
                "val_size": val_size,
                "notes": "trained only on clean strict true-use success sessions",
            }
            torch.save(ckpt, out_path)
            print(f"[strict_video_bc] saved best: {out_path}")

    print(f"[strict_video_bc] done best_val={best_val:.5f}")
    print(f"[strict_video_bc] checkpoint={out_path}")


if __name__ == "__main__":
    main()
