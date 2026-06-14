import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"


VALID_SKILLS = {
    "navigation",
    "wall_avoidance",
    "combat",
    "goal_progress",
    "stuck_behavior",
    "use_action",
    "item_collection",
}

VALID_QUALITIES = {"good", "bad", "mixed"}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="human", choices=["human", "agent"])
    parser.add_argument("--map", default="freedoom1_e1m1")
    parser.add_argument("--skill", required=True, choices=sorted(VALID_SKILLS))
    parser.add_argument("--quality", required=True, choices=sorted(VALID_QUALITIES))
    parser.add_argument("--problem", default="none")
    parser.add_argument("--goal-progress", default="unknown")
    parser.add_argument("--wall-contact", action="store_true")
    parser.add_argument("--combat-state", default="no_enemy")
    parser.add_argument("--preferred-action", default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--frames-dir", required=True, help="Folder containing frames for this clip.")

    args = parser.parse_args()

    src_frames = Path(args.frames_dir)
    if not src_frames.exists():
        raise SystemExit(f"frames-dir does not exist: {src_frames}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    clip_id = datetime.now().strftime("clip_%Y%m%d_%H%M%S")
    clip_dir = OUT_ROOT / clip_id
    dst_frames = clip_dir / "frames"
    dst_frames.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(src_frames.glob("*.png")) +
        list(src_frames.glob("*.jpg")) +
        list(src_frames.glob("*.jpeg"))
    )

    if not image_files:
        raise SystemExit(f"No image files found in {src_frames}")

    for i, src in enumerate(image_files):
        dst = dst_frames / f"frame_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    label = {
        "clip_id": clip_id,
        "source": args.source,
        "map": args.map,
        "skill": args.skill,
        "quality": args.quality,
        "problem": args.problem,
        "goal_progress": args.goal_progress,
        "wall_contact": bool(args.wall_contact),
        "combat_state": args.combat_state,
        "preferred_action": args.preferred_action,
        "notes": args.notes,
        "num_frames": len(image_files),
    }

    with (clip_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(label, f, indent=2)

    print(f"Created skill clip: {clip_dir}")
    print(json.dumps(label, indent=2))


if __name__ == "__main__":
    main()
