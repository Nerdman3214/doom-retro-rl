import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_ROOT = ROOT / "vision_dataset_v2" / "teacher_rollouts_balanced"

counts = Counter()

for metadata_path in sorted(ROLLOUT_ROOT.glob("episode_*/metadata.jsonl")):
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            action_name = row.get("action_name")
            action_index = row.get("action_index")

            counts[(action_index, action_name)] += 1

total = sum(counts.values())

print(f"[teacher_distribution] total={total}")

for (action_index, action_name), count in counts.most_common():
    pct = 100.0 * count / max(total, 1)
    print(
        f"index={action_index} "
        f"name={action_name} "
        f"count={count} "
        f"pct={pct:.2f}%"
    )
