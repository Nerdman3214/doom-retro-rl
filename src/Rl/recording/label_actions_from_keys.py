"""
Smart action labeling system: converts raw keyboard inputs to clean action labels.

Key insight: Distinguish movement actions by looking at KEY COMBINATIONS, not just pressed keys.
- Left/Right arrow keys alone = TURN (rotate in place)
- A/D keys alone = STRAFE (sidestep)
- W/S alone = FORWARD/BACKWARD (move)
- Ctrl/Space = SHOOT/USE (actions)

Usage:
    python label_actions_from_keys.py <input_csv> <output_jsonl>
    
    Input CSV format: frame_path, pressed_keys (comma-separated)
    Output JSONL format: {frame_path, action_name, action_index}
"""

import argparse
import csv
import json
from pathlib import Path


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

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}


def keys_to_action(pressed_keys_str):
    """
    Convert pressed keys (comma-separated string) to single action.
    
    Rules:
    1. Check for exclusive actions first (shoot, use)
    2. Then check movement (movement + shooting combos are ignored for now)
    3. Return the STRONGEST action detected
    
    Priority (highest to lowest):
    - shoot/use (exclusive, no movement)
    - forward/backward movement
    - turn left/right (arrow keys)
    - strafe left/right (a/d without direction keys)
    - nothing (no-op)
    """
    if not pressed_keys_str or pressed_keys_str.strip() == "":
        return None
    
    keys = set(k.strip().lower() for k in pressed_keys_str.split(","))
    
    # Remove weapon selection keys - they don't map to actions
    keys.discard("weapon_1")
    keys.discard("weapon_2")
    keys.discard("weapon_3")
    keys.discard("weapon_4")
    keys.discard("weapon_5")
    keys.discard("weapon_6")
    keys.discard("weapon_7")
    
    has_w = "w" in keys
    has_s = "s" in keys
    has_a = "a" in keys
    has_d = "d" in keys
    has_left = "left" in keys
    has_right = "right" in keys
    has_ctrl = "ctrl" in keys
    has_space = "space" in keys
    
    # Priority 1: Exclusive actions (no movement)
    if has_ctrl and not (has_w or has_s or has_a or has_d or has_left or has_right):
        return "shoot"
    
    if has_space and not (has_w or has_s or has_a or has_d or has_left or has_right):
        return "use"
    
    # Priority 2: Forward/Backward (primary movement)
    if has_w and not has_s:
        return "move_forward"
    if has_s and not has_w:
        return "move_backward"
    
    # Priority 3: Turn (arrow keys exclusive)
    # Skip if conflicting directions pressed simultaneously
    if has_left and not has_right and not has_a and not has_d:
        return "turn_left"
    if has_right and not has_left and not has_a and not has_d:
        return "turn_right"
    if has_left and has_right:
        return None  # Conflicting input, skip
    
    # Priority 4: Strafe (a/d without direction arrows)
    if has_a and not has_d and not has_left and not has_right:
        return "strafe_left"
    if has_d and not has_a and not has_left and not has_right:
        return "strafe_right"
    
    # Priority 5: Complex combos - prefer movement over turning
    if has_w:
        return "move_forward"
    if has_s:
        return "move_backward"
    if has_a:
        return "strafe_left"
    if has_d:
        return "strafe_right"
    if has_left:
        return "turn_left"
    if has_right:
        return "turn_right"
    
    return None


def label_csv_to_jsonl(input_csv, output_jsonl, validate=False):
    """
    Convert CSV (frames + raw keys) to JSONL (frames + action labels).
    
    Args:
        input_csv: Path to input CSV with columns [frame_path, pressed_keys]
        output_jsonl: Path to output JSONL file
        validate: If True, print every 50th sample for manual review
    """
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    no_op_count = 0
    ambiguous_count = 0
    
    with open(input_csv, 'r') as infile, open(output_jsonl, 'w') as outfile:
        reader = csv.DictReader(infile)
        
        for row_idx, row in enumerate(reader):
            frame_path = row.get('frame_path') or row.get('image_path') or list(row.values())[0]
            pressed_keys = row.get('pressed_keys') or row.get('keys') or list(row.values())[1]
            
            action = keys_to_action(pressed_keys)
            
            if action is None:
                no_op_count += 1
                continue  # Skip no-ops
            
            action_index = ACTION_TO_INDEX[action]
            
            output_row = {
                "frame_path": frame_path,
                "action_name": action,
                "action_index": action_index,
                "raw_keys": pressed_keys,
            }
            
            outfile.write(json.dumps(output_row) + "\n")
            valid_count += 1
            
            # Validation sample every 50 frames
            if validate and valid_count % 50 == 0:
                print(f"[Sample {valid_count}] keys='{pressed_keys}' -> action='{action}'")
    
    print(f"\n=== Labeling Summary ===")
    print(f"Valid samples: {valid_count}")
    print(f"No-op frames: {no_op_count}")
    print(f"Total processed: {valid_count + no_op_count}")
    print(f"Saved to: {output_jsonl}")
    
    return valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw keyboard recordings to action labels"
    )
    parser.add_argument("input_csv", help="Input CSV with frame_path and pressed_keys columns")
    parser.add_argument("output_jsonl", help="Output JSONL file")
    parser.add_argument("--validate", action="store_true", help="Print sample labels for review")
    
    args = parser.parse_args()
    
    label_csv_to_jsonl(args.input_csv, args.output_jsonl, validate=args.validate)


if __name__ == "__main__":
    main()
