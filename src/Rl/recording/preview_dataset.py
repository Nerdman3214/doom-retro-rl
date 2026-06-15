"""
Preview tool: Show what your recordings will teach the model BEFORE training.

Usage:
    python preview_dataset.py <dataset_path>
    
This shows:
- Action distribution (% of each action in your data)
- Which actions are rare (model struggles with these)
- Whether there's bias in your playstyle
- Recommendations for what to re-record
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


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


def preview_jsonl_dataset(jsonl_path):
    """
    Show action distribution in JSONL dataset.
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        return
    
    action_counts = defaultdict(int)
    total = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                action = data.get("action_name")
                if action:
                    action_counts[action] += 1
                    total += 1
            except:
                pass
    
    if total == 0:
        print("ERROR: No valid samples found in dataset")
        return
    
    print(f"\n{'='*60}")
    print(f"Dataset: {jsonl_path.name}")
    print(f"Total samples: {total}")
    print(f"{'='*60}\n")
    
    # Sort by count (descending)
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    
    print("Action Distribution (what the model will learn):\n")
    for action, count in sorted_actions:
        pct = count / total * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"{action:15s} {count:4d} ({pct:5.1f}%) {bar}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("Analysis:")
    print(f"{'='*60}\n")
    
    # Find rare actions
    rare_threshold = total * 0.03  # Less than 3%
    rare_actions = [action for action, count in action_counts.items() if count < rare_threshold]
    
    if rare_actions:
        print("⚠️  RARE ACTIONS (model will struggle with these):")
        for action in rare_actions:
            count = action_counts[action]
            pct = count / total * 100
            print(f"   - {action}: {count} samples ({pct:.1f}%)")
        print()
    
    # Find dominant actions
    dominant = [action for action, count in sorted_actions[:3]]
    print(f"✓ STRONG ACTIONS (model learned well):")
    for action in dominant:
        count = action_counts[action]
        pct = count / total * 100
        print(f"   - {action}: {count} samples ({pct:.1f}%)")
    print()
    
    # Recommendations
    print("📋 RECOMMENDATIONS:\n")
    
    if "use" in rare_actions:
        print("   1. Record more gameplay using doors (E key)")
        print("      → Play in areas with locked doors")
        print("      → Open every door you encounter")
        print()
    
    if "move_backward" in rare_actions:
        print("   2. Record moving backward more often (S key)")
        print("      → Back away from enemies")
        print("      → Practice strafing+backing")
        print()
    
    if "turn_left" in rare_actions or "turn_right" in rare_actions:
        print("   3. Record more turning (← → arrow keys)")
        print("      → Turn without moving forward")
        print("      → Practice pure rotation")
        print()
    
    print("   TARGET: Each action should be at least 5-10% of data")
    print("           (Current imbalance is why your model struggles)\n")


def preview_csv_dataset(csv_path):
    """
    Show action distribution in CSV dataset (raw keys).
    """
    from label_actions_from_keys import keys_to_action
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return
    
    action_counts = defaultdict(int)
    total = 0
    
    import csv as csvlib
    
    with open(csv_path, 'r') as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            pressed_keys = row.get('pressed_keys') or row.get('keys')
            if pressed_keys:
                action = keys_to_action(pressed_keys)
                if action:
                    action_counts[action] += 1
                    total += 1
    
    if total == 0:
        print("ERROR: No valid samples found in CSV")
        return
    
    print(f"\n{'='*60}")
    print(f"CSV Dataset: {csv_path.name}")
    print(f"Total samples: {total}")
    print(f"{'='*60}\n")
    
    # Sort by count (descending)
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    
    print("Raw Key Action Distribution:\n")
    for action, count in sorted_actions:
        pct = count / total * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"{action:15s} {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\nReady to label? Run:")
    print(f"  python label_actions_from_keys.py {csv_path} output.jsonl --validate")


def main():
    if len(sys.argv) < 2:
        print("Usage: python preview_dataset.py <path_to_jsonl_or_csv>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if dataset_path.endswith('.jsonl'):
        preview_jsonl_dataset(dataset_path)
    elif dataset_path.endswith('.csv'):
        preview_csv_dataset(dataset_path)
    else:
        print("ERROR: File must be .jsonl or .csv")


if __name__ == "__main__":
    main()
