"""
Convert recorded gameplay CSV → JSONL dataset → add to training folder.

Complete pipeline:
  1. Record with: python record_doomretro_screen_keys.py <name>
     → Saves CSV + frames to vision_dataset_v2/doomretro_keyed_imitation/<name>/
  
  2. Label with: python label_actions_from_keys.py <csv> <jsonl> --validate
     → Preview action distribution, confirm it looks good
  
  3. Add to dataset with: python add_to_supervised_dataset.py <jsonl> <name>
     → Creates episode in vision_dataset_v2/teacher_rollouts_balanced/episode_XXXXX/
     → Ready for training!
  
  4. Train with: PYTHONPATH=. python supervised/train_human_policy.py --epochs 5 --enable-augment --oversample 2.0
     → Model learns from all episodes combined

Example full workflow:
    # Record gameplay session
    python record_doomretro_screen_keys.py session_01
    
    # Label the actions
    python label_actions_from_keys.py vision_dataset_v2/doomretro_keyed_imitation/session_01/actions.csv /tmp/session_01.jsonl --validate
    
    # Preview what model will learn
    python preview_dataset.py /tmp/session_01.jsonl
    
    # Add to training dataset
    python add_to_supervised_dataset.py /tmp/session_01.jsonl session_01
    
    # Retrain
    cd ..
    PYTHONPATH=. python supervised/train_human_policy.py --epochs 5 --enable-augment --oversample 2.0 --pos-weight-mode clamped
"""

import argparse
import json
import shutil
from pathlib import Path


def add_recording_to_dataset(jsonl_path, recording_name, dataset_root="vision_dataset_v2/teacher_rollouts_balanced"):
    """
    Add a validated JSONL recording to the training dataset.
    
    Creates a new episode folder with metadata.jsonl file.
    """
    jsonl_path = Path(jsonl_path)
    dataset_root = Path(dataset_root)
    
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        return False
    
    # Find next episode number
    episode_dirs = sorted(dataset_root.glob("episode_*"))
    if episode_dirs:
        last_num = int(episode_dirs[-1].name.split("_")[1])
        next_num = last_num + 1
    else:
        next_num = 1
    
    episode_name = f"episode_{next_num:06d}"
    episode_dir = dataset_root / episode_name
    
    # Create episode directory
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy JSONL as metadata.jsonl
    metadata_file = episode_dir / "metadata.jsonl"
    shutil.copy(jsonl_path, metadata_file)
    
    # Count samples
    sample_count = 0
    with open(metadata_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    print(f"\n{'='*60}")
    print(f"✓ Added to dataset: {episode_name}")
    print(f"{'='*60}")
    print(f"Location: {metadata_file}")
    print(f"Samples:  {sample_count}")
    print(f"\nNext step:")
    print(f"  cd /home/steven/Downloads/doomretro-master/src/Rl")
    print(f"  PYTHONPATH=. python supervised/train_human_policy.py \\")
    print(f"    --epochs 5 \\")
    print(f"    --enable-augment \\")
    print(f"    --oversample 2.0 \\")
    print(f"    --pos-weight-mode clamped")
    print()
    
    return True


def list_episodes():
    """Show all current episodes in dataset."""
    dataset_root = Path("vision_dataset_v2/teacher_rollouts_balanced")
    
    if not dataset_root.exists():
        print("No dataset found yet")
        return
    
    episodes = sorted(dataset_root.glob("episode_*"))
    
    total_samples = 0
    print(f"\nCurrent dataset episodes:\n")
    
    for ep_dir in episodes:
        metadata = ep_dir / "metadata.jsonl"
        if metadata.exists():
            sample_count = sum(1 for _ in open(metadata))
            total_samples += sample_count
            print(f"  {ep_dir.name}: {sample_count} samples")
    
    print(f"\nTotal: {len(episodes)} episodes, {total_samples} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Add recording to supervised learning dataset"
    )
    parser.add_argument("jsonl_path", help="Path to labeled JSONL file")
    parser.add_argument("recording_name", help="Name of recording (for logging)")
    parser.add_argument("--list", action="store_true", help="List current episodes")
    parser.add_argument("--dataset-root", default="vision_dataset_v2/teacher_rollouts_balanced")
    
    args = parser.parse_args()
    
    if args.list:
        list_episodes()
        return
    
    success = add_recording_to_dataset(args.jsonl_path, args.recording_name, args.dataset_root)
    
    if success:
        list_episodes()


if __name__ == "__main__":
    main()
