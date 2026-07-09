#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import numpy as np


DATA_ROOT = Path("vision_dataset/raw_human_play")
CKPT = Path("checkpoints/vizdoom_video_behavior_clone.pt")

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

SAFE_ROUTE_ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
}


def truthy(v):
    return str(v).strip().lower() in {"1", "1.0", "true", "yes"}


def parse_buttons(row):
    """Return multi-hot button labels.

    Supports both formats:
      1. action column: move_forward+turn_left
      2. separate button columns: move_forward=1, turn_left=1, etc.
    """
    action_text = (
        row.get("action")
        or row.get("active")
        or row.get("buttons")
        or ""
    )

    action_text = str(action_text).strip().lower()

    tokens = set()
    for part in (
        action_text
        .replace(",", "+")
        .replace("|", "+")
        .replace(";", "+")
        .replace(" ", "+")
        .split("+")
    ):
        part = part.strip().lower()
        if part and part not in ("no_op", "noop", "none", "0"):
            tokens.add(part)

    values = []

    for name in ACTION_NAMES:
        raw = (
            row.get(name)
            or row.get(name.upper())
            or row.get(name.replace("_", "-"))
            or ""
        )

        raw_text = str(raw).strip().lower()

        on_from_column = raw_text in ("1", "true", "yes", "y", "on", "pressed")
        on_from_action = name in tokens

        values.append(1.0 if (on_from_column or on_from_action) else 0.0)

    return torch.tensor(values, dtype=torch.float32)


def is_safe_route_action(y):
    """Return True for simple route-following actions only."""
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().tolist()

    if not isinstance(y, (list, tuple)) or len(y) != len(ACTION_NAMES):
        return False

    active = [name for name, value in zip(ACTION_NAMES, y) if float(value) >= 0.5]
    return bool(active) and set(active).issubset(SAFE_ROUTE_ACTIONS)


class VideoDoomDataset(Dataset):
    def __init__(
        self,
        seq_len=8,
        image_size=84,
        min_rows=80,
        data_root=None,
        frame_cache_size=128,
        skip_broken=True,
        max_sessions=0,
    ):
        self.seq_len = seq_len
        self.samples = []
        self.image_size = image_size
        self.data_root = Path(data_root or DATA_ROOT)
        self.frame_cache = {}
        self.frame_cache_size = max(0, frame_cache_size)
        self.skip_broken = skip_broken

        sessions = sorted(self.data_root.glob("session_*/actions.csv"))
        if max_sessions > 0:
            sessions = sessions[:max_sessions]
        print(f"[video_bc] found sessions: {len(sessions)}")

        skipped = []
        action_counts = Counter()

        for csv_path in sessions:
            session = csv_path.parent
            frames_dir = session / "frames"

            if not frames_dir.exists():
                skipped.append((session.name, "missing frames dir"))
                continue

            try:
                rows = list(csv.DictReader(csv_path.open()))
            except Exception as exc:
                skipped.append((session.name, f"csv_error={exc}"))
                continue

            if len(rows) < max(min_rows, seq_len + 1):
                skipped.append((session.name, f"rows={len(rows)}"))
                continue

            usable = []
            for row in rows:
                frame_id = row.get("frame", "")
                try:
                    frame_id = int(float(frame_id))
                except Exception:
                    continue

                frame_path = frames_dir / f"frame_{frame_id:06d}.png"
                if not frame_path.exists():
                    frame_path = frames_dir / f"frame_{frame_id}.png"
                if not frame_path.exists():
                    if self.skip_broken:
                        continue
                    raise FileNotFoundError(
                        f"Missing frame for {session.name}: {frame_path}"
                    )

                y = parse_buttons(row)
                if not is_safe_route_action(y):
                    continue
                usable.append((frame_path, y, row.get("action", "no_op")))
                action_counts[row.get("action", "no_op")] += 1

            if len(usable) < max(min_rows, seq_len + 1):
                skipped.append((session.name, f"usable={len(usable)}"))
                continue

            for end in range(seq_len - 1, len(usable)):
                frame_paths = [
                    usable[i][0] for i in range(end - seq_len + 1, end + 1)
                ]
                y = usable[end][1]
                self.samples.append((frame_paths, y))

        print(f"[video_bc] samples: {len(self.samples)}")
        print("[video_bc] top actions:")
        for k, v in action_counts.most_common(20):
            print(f"  {k:35} {v}")

        if skipped:
            print("[video_bc] skipped:")
            for name, reason in skipped[:30]:
                print(f"  {name}: {reason}")

        if not self.samples:
            raise RuntimeError(f"No video samples found in {DATA_ROOT}")

    def __len__(self):
        return len(self.samples)

    def _load_frame_tensor(self, p):
        cache_key = str(p)
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        try:
            img = Image.open(p).convert("RGB").resize(
                (self.image_size, self.image_size),
                Image.BILINEAR,
            )
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        frame = torch.from_numpy(arr).permute(2, 0, 1)
        if self.frame_cache_size > 0:
            if len(self.frame_cache) >= self.frame_cache_size:
                self.frame_cache.pop(next(iter(self.frame_cache)))
            self.frame_cache[cache_key] = frame
        return frame

    def __getitem__(self, idx):
        frame_paths, y = self.samples[idx]
        frames = [self._load_frame_tensor(p) for p in frame_paths]
        x = torch.stack(frames, dim=0)  # T,C,H,W
        return x, y


class VideoBCNet(nn.Module):
    def __init__(self, num_actions=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        # x: B,T,C,H,W
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.cnn(x).flatten(1)
        feat = feat.reshape(b, t, -1)
        out, _ = self.gru(feat)
        last = out[:, -1]
        return self.head(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=84)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--min-rows", type=int, default=80)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--data-root", default="vision_dataset/raw_human_play")
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--frame-cache-size", type=int, default=128)
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument(
        "--skip-broken",
        dest="skip_broken",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-skip-broken",
        dest="skip_broken",
        action="store_false",
    )
    args = ap.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[video_bc] device:", device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = VideoDoomDataset(
        seq_len=args.seq_len,
        image_size=args.image_size,
        min_rows=args.min_rows,
        data_root=args.data_root,
        frame_cache_size=args.frame_cache_size,
        skip_broken=args.skip_broken,
        max_sessions=args.max_sessions,
    )

    val_size = max(1, int(len(dataset) * 0.10))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VideoBCNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    CKPT.parent.mkdir(parents=True, exist_ok=True)

    best_val = 999.0

    scaler = torch.amp.GradScaler() if args.amp and device == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            train_loss += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        exact = 0
        total = 0
        button_correct = 0
        button_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()

                val_loss += float(loss.item())
                val_batches += 1

                exact += (pred == y).all(dim=1).sum().item()
                total += y.shape[0]
                button_correct += (pred == y).sum().item()
                button_total += y.numel()

        train_loss /= max(1, train_batches)
        val_loss /= max(1, val_batches)
        exact_acc = exact / max(1, total)
        button_acc = button_correct / max(1, button_total)

        print(
            f"[video_bc] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"exact={exact_acc:.3f} button_acc={button_acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "seq_len": args.seq_len,
                "image_size": args.image_size,
                "action_names": ACTION_NAMES,
            }, CKPT)
            print(f"[video_bc] saved {CKPT}")

    print("[video_bc] done")


if __name__ == "__main__":
    main()
