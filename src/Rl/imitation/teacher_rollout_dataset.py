import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class TeacherRolloutDataset(Dataset):
    """
    Loads ViZDoom teacher rollout data.

    Each item:
        image tensor: [3, H, W], float32, 0..1
        action index: long
    """

    def __init__(self, rollout_root, image_size=(84, 84)):
        self.rollout_root = Path(rollout_root)
        self.image_size = image_size
        self.samples = []

        metadata_files = sorted(self.rollout_root.glob("episode_*/metadata.jsonl"))

        for metadata_path in metadata_files:
            episode_dir = metadata_path.parent

            with metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    row = json.loads(line)

                    frame_rel = row.get("frame")
                    action_index = row.get("action_index")

                    if frame_rel is None or action_index is None:
                        continue

                    frame_path = episode_dir / frame_rel

                    if not frame_path.exists():
                        continue

                    self.samples.append(
                        {
                            "frame_path": frame_path,
                            "action_index": int(action_index),
                            "action_name": row.get("action_name"),
                            "reward": float(row.get("reward", 0.0)),
                        }
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No teacher rollout samples found in {self.rollout_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(str(sample["frame_path"]), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {sample['frame_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1).contiguous()

        action = torch.tensor(sample["action_index"], dtype=torch.long)

        return image, action
