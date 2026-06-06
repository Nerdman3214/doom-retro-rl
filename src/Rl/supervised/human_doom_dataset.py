import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_DIM = 8


class HumanDoomDataset(Dataset):
    """
    Supervised dataset for Doom human gameplay.

    Input:
      image frame

    Target:
      8-button vector:
        0 move_forward
        1 move_backward
        2 turn_left
        3 turn_right
        4 strafe_left
        5 strafe_right
        6 shoot
        7 use

    This supports compound actions:
      move_forward+turn_left
      strafe_right+shoot
    """

    def __init__(self, dataset_root, image_size=84, max_no_op_ratio=0.20):
        self.dataset_root = Path(dataset_root)
        self.image_size = int(image_size)
        self.max_no_op_ratio = float(max_no_op_ratio)

        self.samples = []
        self._load_samples()

    def _find_csvs(self):
        csvs = sorted(self.dataset_root.glob("session_*/actions.csv"))
        csvs += sorted(self.dataset_root.glob("actions.csv"))
        return csvs

    def _parse_buttons(self, raw):
        if not raw:
            return None

        parts = str(raw).replace(",", " ").split()
        values = [int(float(x)) for x in parts]

        if len(values) != ACTION_DIM:
            return None

        return values

    def _load_samples(self):
        csvs = self._find_csvs()

        if not csvs:
            raise FileNotFoundError(f"No actions.csv files found under {self.dataset_root}")

        no_op_samples = []
        action_samples = []

        for csv_path in csvs:
            session_dir = csv_path.parent

            with csv_path.open() as f:
                reader = csv.DictReader(f)

                for row in reader:
                    frame_rel = row.get("frame_path", "")
                    if not frame_rel:
                        continue

                    frame_path = session_dir / frame_rel
                    if not frame_path.exists():
                        continue

                    buttons = self._parse_buttons(row.get("buttons", ""))
                    if buttons is None:
                        continue

                    sample = {
                        "frame_path": frame_path,
                        "buttons": np.asarray(buttons, dtype=np.float32),
                        "action": row.get("action", ""),
                    }

                    if sum(buttons) == 0:
                        no_op_samples.append(sample)
                    else:
                        action_samples.append(sample)

        # Keep no_op from dominating.
        max_no_op = int(len(action_samples) * self.max_no_op_ratio)
        no_op_samples = no_op_samples[:max_no_op]

        self.samples = action_samples + no_op_samples

        if not self.samples:
            raise RuntimeError("Dataset loaded zero usable samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(str(sample["frame_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {sample['frame_path']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Convert to CHW float tensor.
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img)
        y = torch.from_numpy(sample["buttons"])

        return x, y
