# Option C: RL Fine-Tuning From Imitation

Current best base policy:
- checkpoints/doomretro_keymouse_sequence_cnn.pt

Current live controller:
- tools/run_doomretro_keymouse_sequence_agent.py

Goal:
- Do not train RL from scratch.
- Use imitation for low-level keyboard/mouse control.
- Train RL as a high-level route/recovery policy.

Macro actions:
0. trust_bc
1. force_forward
2. forward_strafe_left
3. forward_strafe_right
4. forward_turn_left
5. forward_turn_right
6. tap_use_forward
7. combat_forward

Reward shaping:
+ enemy killed
+ forward progress
+ new area / exploration
+ route progress toward exit
+ successful use/open
+ level exit

Penalties:
- repeated no_op
- wall stuck
- repeated 180 turns
- useless use spam
- pure strafe loop
- health loss
- death

Training order:
1. Keep current BC sequence model frozen.
2. Train high-level macro policy in sim.
3. Use macro policy to bias/override live BC agent.
4. Later fine-tune the BC model itself if needed.
