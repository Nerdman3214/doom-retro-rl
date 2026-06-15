import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--suffix", default="split")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    frames_dir = clip_dir / "frames"
    labels_path = clip_dir / "labels.json"

    if not frames_dir.exists():
        raise SystemExit(f"Missing frames folder: {frames_dir}")
    if not labels_path.exists():
        raise SystemExit(f"Missing labels.json: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        base_label = json.load(f)

    frames = sorted(
        list(frames_dir.glob("*.png")) +
        list(frames_dir.glob("*.jpg")) +
        list(frames_dir.glob("*.jpeg"))
    )

    if not frames:
        raise SystemExit("No frames found.")

    made = 0

    for start in range(0, len(frames), args.stride):
        end = start + args.chunk_size
        chunk = frames[start:end]

        if len(chunk) < max(20, args.chunk_size // 3):
            continue

        new_clip_id = f"{base_label['clip_id']}_{args.suffix}_{made:04d}"
        out_dir = DATA_ROOT / new_clip_id
        out_frames = out_dir / "frames"
        out_frames.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(chunk):
            dst = out_frames / f"frame_{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        label = dict(base_label)
        label["clip_id"] = new_clip_id
        label["parent_clip_id"] = base_label["clip_id"]
        label["start_frame"] = start
        label["end_frame"] = start + len(chunk) - 1
        label["num_frames"] = len(chunk)
        label["notes"] = base_label.get("notes", "") + " | auto-split short clip"

        with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
            json.dump(label, f, indent=2)

        made += 1

    print(f"Created {made} short clips from {clip_dir}")


if __name__ == "__main__":
    main()
