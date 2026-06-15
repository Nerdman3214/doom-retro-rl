# Re-Recording Guide for Supervised Learning Dataset Improvement

This guide provides a structured plan to record additional human gameplay data to improve the supervised learning policy model, especially for underrepresented actions.

## Current Dataset Status (After A-B-C Improvements)

After applying threshold calibration, oversampling, and data augmentation, your model achieves:
- **Overall accuracy: 75.4%** (test set with optimal thresholds)
- **Per-action performance:**
  - move_forward: F1=0.736
  - move_backward: F1=0.760 ← Relatively weak
  - turn_left: F1=0.770
  - turn_right: F1=0.866 ← Good
  - strafe_left: F1=0.898
  - strafe_right: F1=0.937 ← Excellent
  - shoot: F1=0.788
  - **use: F1=0.994** ← Now excellent due to oversampling!

## Re-Recording Strategy

The current model is strong overall, but you can still improve the weaker actions:

### Priority Actions to Record
1. **move_backward** (F1=0.760) — Low priority, decent performance
2. **move_forward** (F1=0.736) — Low-medium priority
3. All others — Optional, already good

### Total Data Collection Target
- **Baseline:** You currently have ~3,200 samples
- **Target:** 5,000–6,000 total samples (50–100% increase)
- **New recording:** ~2,000–3,000 new samples

### Breakdown per Action
For the weakest actions, aim for:
- **move_backward:** 200–300 new samples (~2 min of continuous backward movement at typical play speed)
- **move_forward:** 300–400 new samples (~3 min of continuous forward movement)
- **Other actions (optional):** Distribute remaining quota as natural gameplay

## How to Record

### Step 1: Use Existing Recording Tools
Run the behavior logger in your Doom environment:
```bash
cd src/Rl
PYTHONPATH=. python -c "
from loggers.behavior_logger import BehaviorLogger
logger = BehaviorLogger(output_dir='vision_dataset_v2/teacher_rollouts_focused_rerecord')
# Start your game loop with this logger attached
# Each episode is automatically saved to the dataset
"
```

Or use your existing replay mechanism (if available):
```bash
cd src/Rl
PYTHONPATH=. python tools/run_doomretro_keymouse_sequence_agent.py --record-dataset vision_dataset_v2/teacher_rollouts_focused_rerecord
```

### Step 2: Focus Sessions

#### Session A: move_backward Focused (10–15 minutes)
- Spawn in an open area (e.g., MAP01)
- Hold backspace or 's' key to move backward continuously
- Vary movement: strafe while moving backward, turn while moving backward
- **Target:** 200–300 backward movement samples

#### Session B: move_forward Focused (10–15 minutes)
- Navigate the map while holding forward movement key ('w')
- Include natural turning, strafing, and shooting while moving forward
- **Target:** 300–400 forward movement samples

#### Session C: Mixed Natural Play (10 minutes, optional)
- Play naturally to collect balanced multi-action sequences
- Let the system capture whatever emerges

### Step 3: Combine Datasets

After recording, merge your new data with existing data:
```bash
cd src/Rl
python -c "
import shutil
from pathlib import Path

# Copy new episodes from focused recording to main dataset
src = Path('vision_dataset_v2/teacher_rollouts_focused_rerecord')
dst = Path('vision_dataset_v2/teacher_rollouts_balanced')

for episode_dir in src.glob('episode_*'):
    shutil.copytree(episode_dir, dst / episode_dir.name)

print('Merged datasets. New size:')
import os
total = len(list(dst.glob('episode_*/metadata.jsonl')))
print(f'Total episodes: {total}')
"
```

### Step 4: Re-Train Model

Re-train with the merged dataset:
```bash
cd src/Rl
PYTHONPATH=. python supervised/train_human_policy.py \
  --epochs 10 \
  --batch-size 64 \
  --enable-augment \
  --oversample 2.0 \
  --pos-weight-mode clamped
```

## Expected Outcomes

**After 500–1000 new backward+forward samples:**
- move_backward F1: 0.760 → ~0.800–0.820
- move_forward F1: 0.736 → ~0.780–0.800
- Overall accuracy: 75.4% → ~78–80%

**After 2000–3000 new balanced samples:**
- move_backward F1: 0.760 → ~0.850+
- move_forward F1: 0.736 → ~0.850+
- Overall accuracy: 75.4% → ~82–85%

## Alternative: Low-Cost Improvements First

Before re-recording, consider:

1. **Longer training (more epochs):**
   ```bash
   python supervised/train_human_policy.py --epochs 20 --enable-augment --oversample 2.0
   ```
   Expected boost: +2–3% accuracy.

2. **Stronger augmentation:**
   - Modify `_augment_image()` in `human_doom_dataset.py` to add:
     - Gaussian noise
     - More aggressive rotation (±10 degrees)
     - Temporal frame stacking (use previous frame to capture motion context)

3. **Curriculum learning:**
   - Start with easy classes (move_forward, shoot)
   - Gradually add hard classes (move_backward)
   - Implement by sorting training samples by action rarity

## Troubleshooting

**Issue:** "No new episodes recorded"
- Verify game is running and frames are being captured
- Check output directory permissions
- Look at `trajectory_logger.py` or `behavior_logger.py` for frame save logic

**Issue:** New data doesn't improve model
- Dataset may have label errors (check frame-to-action alignment)
- New recordings may have different visual style (lighting, map differences)
  - Solution: Normalize recordings to same maps/lighting as original
- Action detection may be miscalibrated
  - Verify button mapping in recorder vs. model expectations

**Issue:** Model performance gets worse
- Data distribution mismatch: new data differs significantly from original
  - Solution: Pre-train on original data, fine-tune on mixed data
- Oversampling too aggressive: 2x factor may be hurting general actions
  - Solution: Reduce to `--oversample 1.5`

## Next Steps

1. **Short-term:** Run A+B+C improvements (✓ done) and evaluate
2. **Medium-term:** If still unsatisfied, record 500–1000 focused samples
3. **Long-term:** Integrate trained policy as RL prior or curriculum-based action generator

---

**Contact / Questions?** See [behavior_logger.py](./loggers/behavior_logger.py) for dataset format details.
