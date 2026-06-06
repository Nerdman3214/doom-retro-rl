import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"


def main():
    labels = sorted(DATA_ROOT.glob("clip_*/labels.json"))

    print(f"[skill_clips] root={DATA_ROOT}")
    print(f"[skill_clips] total_label_files={len(labels)}")

    skill_counts = Counter()
    quality_counts = Counter()
    problem_counts = Counter()
    source_counts = Counter()
    skill_quality_counts = Counter()
    frame_counts = []

    examples = defaultdict(list)

    for path in labels:
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)

        skill = item.get("skill", "unknown")
        quality = item.get("quality", "unknown")
        problem = item.get("problem", "unknown")
        source = item.get("source", "unknown")
        num_frames = int(item.get("num_frames", 0))

        skill_counts[skill] += 1
        quality_counts[quality] += 1
        problem_counts[problem] += 1
        source_counts[source] += 1
        skill_quality_counts[(skill, quality)] += 1
        frame_counts.append(num_frames)

        key = f"{skill}/{quality}"
        if len(examples[key]) < 3:
            examples[key].append(str(path.parent.relative_to(DATA_ROOT)))

    print("\n[by source]")
    for k, v in source_counts.most_common():
        print(f"  {k}: {v}")

    print("\n[by skill]")
    for k, v in skill_counts.most_common():
        print(f"  {k}: {v}")

    print("\n[by quality]")
    for k, v in quality_counts.most_common():
        print(f"  {k}: {v}")

    print("\n[by problem]")
    for k, v in problem_counts.most_common():
        print(f"  {k}: {v}")

    print("\n[by skill + quality]")
    for (skill, quality), v in skill_quality_counts.most_common():
        print(f"  {skill}/{quality}: {v}")

    if frame_counts:
        print("\n[frames]")
        print(f"  min={min(frame_counts)}")
        print(f"  max={max(frame_counts)}")
        print(f"  avg={sum(frame_counts) / len(frame_counts):.1f}")

    print("\n[examples]")
    for key, vals in examples.items():
        print(f"  {key}:")
        for v in vals:
            print(f"    {v}")


if __name__ == "__main__":
    main()
