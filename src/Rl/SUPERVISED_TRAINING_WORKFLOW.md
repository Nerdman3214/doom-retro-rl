"""
SUPERVISED LEARNING IMPROVEMENT GUIDE
=====================================

Your Issue: Model doesn't improve because good recordings aren't being properly labeled/integrated.
Your Goal: Complete E1M1 → find secrets → kill all enemies in corridor route

NEW WORKFLOW: Record → Label → Preview → Add → Train

This guide shows you how to add high-quality recordings to your dataset and improve the model.

================================================================================
STEP 1: RECORD CLEAN GAMEPLAY
================================================================================

IMPORTANT: Focus on the exact route you want the model to learn:
  1. Kill enemy in corridor
  2. Exit corridor
  3. Kill enemy next to non-blue key door
  4. Enter door
  5. Kill all enemies inside
  6. Go to next door
  7. Hit exit button

Command:
  cd /home/steven/Downloads/doomretro-master/src/Rl
  python recording/record_doomretro_screen_keys.py session_01

Then:
  1. Launch Doom Retro manually: ./build/doomretro -iwad /usr/share/games/doom/freedoom1.wad
  2. Select: New Game → Easiest Difficulty → E1M1
  3. Focus the Doom window
  4. Press ENTER in the recording terminal
  5. Play ONE COMPLETE RUN (start to exit)
  6. Press Ctrl+C in terminal when done

Output: vision_dataset_v2/doomretro_keyed_imitation/session_01/
  - frames/ folder (PNG images)
  - actions.csv (frame_path, pressed_keys columns)

Size estimate: ~100-300 MB per complete run (acceptable)

================================================================================
STEP 2: LABEL THE ACTIONS (Better Than Before!)
================================================================================

Your recordings have raw keyboard data. This converts them to proper action labels
using intelligent key-to-action mapping:

  LEFT/RIGHT arrows = TURN (rotate in place)
  A/D keys = STRAFE (sidestep)
  W/S keys = MOVE FORWARD/BACKWARD
  CTRL = SHOOT
  SPACE/E = USE (open doors)

Command:
  python recording/label_actions_from_keys.py \\
    vision_dataset_v2/doomretro_keyed_imitation/session_01/actions.csv \\
    /tmp/session_01_labeled.jsonl \\
    --validate

Output: /tmp/session_01_labeled.jsonl (small file, ~2-5 MB)

================================================================================
STEP 3: PREVIEW THE DATA (See What Model Will Learn!)
================================================================================

Before training, preview the action distribution to catch problems:

Command:
  python recording/preview_dataset.py /tmp/session_01_labeled.jsonl

Example output:
  ══════════════════════════════════════════════════════════════
  Dataset: session_01_labeled.jsonl
  Total samples: 450
  ══════════════════════════════════════════════════════════════
  
  Action Distribution (what the model will learn):
  
  move_forward    248 ( 55.1%) ██████████████████████████
  shoot           120 ( 26.7%) █████████████
  turn_left        42 (  9.3%) ████
  use              25 (  5.6%) ██
  strafe_left      10 (  2.2%) █
  move_backward     5 (  1.1%)
  ...

✓ GOOD DATA: All actions present, use/shoot visible
✗ BAD DATA: Missing actions (model won't learn them)
✗ BAD DATA: 90% forward movement (unbalanced, model gets stuck)

If data looks bad:
  - Delete and re-record
  - Focus on variety: turning, strafing, using items, not just rushing forward

================================================================================
STEP 4: ADD TO TRAINING DATASET
================================================================================

Command:
  python recording/add_to_supervised_dataset.py \\
    /tmp/session_01_labeled.jsonl \\
    session_01

This creates: vision_dataset_v2/teacher_rollouts_balanced/episode_000002/

Output shows:
  ✓ Added to dataset: episode_000002
  Location: vision_dataset_v2/teacher_rollouts_balanced/episode_000002/metadata.jsonl
  Samples:  450
  
  Current dataset episodes:
    episode_000001: 3195 samples
    episode_000002: 450 samples
  
  Total: 2 episodes, 3645 samples

Now your dataset has: Original 3,195 + your 450 = 3,645 samples total

================================================================================
STEP 5: RETRAIN WITH NEW DATA
================================================================================

Command (from src/Rl/):
  PYTHONPATH=. python supervised/train_human_policy.py \\
    --epochs 10 \\
    --batch-size 32 \\
    --enable-augment \\
    --oversample 2.0 \\
    --pos-weight-mode clamped

This trains on ALL episodes combined (original + your new recordings).

Expected improvement:
  - Before (original data only): ~75% accuracy, bad on rare actions
  - After (original + 1 clean session): ~78-80% accuracy, better on use/turn
  - After (original + 5 clean sessions): ~82-85% accuracy, much better overall

Watch the per-class metrics:
  shoot  p=0.966 r=0.308 f1=0.467  ← Should improve!
  use    p=0.004 r=1.000 f1=0.008  ← Should improve!

================================================================================
BEST PRACTICES FOR HIGH-QUALITY DATA
================================================================================

✓ DO:
  - Record multiple complete runs (variety)
  - Play the exact route every time (consistency)
  - Use all actions: move, turn, strafe, shoot, use doors
  - Play at EASIEST difficulty (consistent gameplay)
  - Focus on the corridor route (your specific goal)
  
✗ DON'T:
  - Record incomplete runs (partial data)
  - Mix different playstyles in one session
  - Record movement-only (no shooting/using items)
  - Record erratic movement (model learns chaos)
  - Record on hard difficulty (inconsistent patterns)

Recording checklist for each session:
  [ ] Started at level beginning
  [ ] Killed corridor enemy
  [ ] Exited corridor properly
  [ ] Used door (tested 'use' action)
  [ ] Completed full route
  [ ] Reached exit button
  [ ] Recorded to completion without crashing

================================================================================
ITERATIVE WORKFLOW
================================================================================

Repeat this cycle:

  Session 1:
    Record → Label → Preview → Add → Train → Evaluate
    
  Review results:
    - If "use" still weak: Record more door-opening
    - If "turn_left" weak: Record more pure turning
    - If "move_backward" weak: Record strafing backwards
    
  Session 2:
    Record (DIFFERENT run) → Label → Preview → Add → Train → Evaluate
    
  Session 3+:
    Keep adding quality sessions until accuracy plateaus

================================================================================
DISK SPACE MANAGEMENT
================================================================================

Frame PNG files are large (~100-300 MB per session).
After labeling + adding to dataset, DELETE the frames folder:

  rm -rf vision_dataset_v2/doomretro_keyed_imitation/session_01/frames/

Keep the labeled JSONL (~2-5 MB each) in dataset:
  vision_dataset_v2/teacher_rollouts_balanced/episode_000002/metadata.jsonl

This way: 100 sessions = ~200-500 MB (JSONL files) instead of 10+ GB (PNG files)

================================================================================
TROUBLESHOOTING
================================================================================

Q: Data preview shows "use" as 0% of samples
A: You didn't press space/E to open doors. Record again, focus on using items.

Q: "move_forward" is 90% of data
A: You're only rushing forward. Record again with more careful movement:
   - Move forward, then TURN to aim
   - Move forward, STOP, SHOOT
   - Use STRAFE (A/D keys) instead of turning

Q: Model accuracy didn't improve after 1 new session
A: Normal! You added 450 samples to 3,195. Need more sessions:
   - 1 session (450): +2% improvement expected
   - 5 sessions (2,250): +5-8% improvement expected
   - 10 sessions (4,500): +10-15% improvement expected

Q: "ValueError: No actions.csv files found"
A: Recording didn't save properly. Check:
   - Did you press Ctrl+C to stop recording?
   - Is the session name spelled correctly?
   - Does folder exist: vision_dataset_v2/doomretro_keyed_imitation/session_01/

================================================================================
NEXT STEPS
================================================================================

1. Record your first session: python recording/record_doomretro_screen_keys.py session_01
2. Label it: python recording/label_actions_from_keys.py ...
3. Preview: python recording/preview_dataset.py /tmp/session_01_labeled.jsonl
4. Add: python recording/add_to_supervised_dataset.py ...
5. Train: PYTHONPATH=. python supervised/train_human_policy.py --epochs 10 --enable-augment --oversample 2.0
6. Test: Load checkpoint and verify model plays better
7. Repeat with sessions 2-5 for bigger improvements

"""
