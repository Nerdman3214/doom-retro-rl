from pathlib import Path
import re

p = Path("recording/record_vizdoom_wall_recovery.py")
text = p.read_text()

backup = Path("recording/record_vizdoom_wall_recovery.py.backup_before_halfwall_traps")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add half-wall / rail-like trap setups. These are still action-sequence based,
# not teleport based. They try to create side-rail/half-wall stuck behavior.
insert = '''
    {
        "name": "half_wall_left_rail_scrape",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 1, 0, 0, 0], 65),
            ([1, 0, 1, 0, 1, 0, 0, 0], 35),
            ([0, 0, 1, 0, 1, 0, 0, 0], 20),
        ],
    },
    {
        "name": "half_wall_right_rail_scrape",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 1, 0, 0], 65),
            ([1, 0, 0, 1, 0, 1, 0, 0], 35),
            ([0, 0, 0, 1, 0, 1, 0, 0], 20),
        ],
    },
    {
        "name": "rail_forward_pressure_left_turn",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 45),
            ([1, 0, 1, 0, 0, 0, 0, 0], 45),
            ([1, 0, 1, 0, 1, 0, 0, 0], 35),
        ],
    },
    {
        "name": "rail_forward_pressure_right_turn",
        "recovery_mode": "half_wall_rail",
        "sequence": [
            ([1, 0, 0, 0, 0, 0, 0, 0], 45),
            ([1, 0, 0, 1, 0, 0, 0, 0], 45),
            ([1, 0, 0, 1, 0, 1, 0, 0], 35),
        ],
    },
'''

marker = "TRAP_SETUPS = ["
if marker not in text:
    raise SystemExit("Could not find TRAP_SETUPS list.")

if "half_wall_left_rail_scrape" not in text:
    text = text.replace(marker, marker + insert, 1)

# Add default recovery_mode to old traps when selected.
text = text.replace(
    "current_trap = reset_and_make_trap(env, game, args.frame_skip)",
    "current_trap, current_recovery_mode = reset_and_make_trap(env, game, args.frame_skip)",
    1,
)

text = text.replace(
    "current_trap = reset_and_make_trap(env, game, args.frame_skip)",
    "current_trap, current_recovery_mode = reset_and_make_trap(env, game, args.frame_skip)",
)

# Change reset function to return both trap name and recovery mode.
text = text.replace(
    "return trap[\"name\"]",
    "return trap[\"name\"], trap.get(\"recovery_mode\", \"wall_recovery\")",
)

# Add recovery_mode to CSV row.
if '"recovery_mode": current_recovery_mode,' not in text:
    text = text.replace(
        '"trap": current_trap,',
        '"trap": current_trap,\n                "recovery_mode": current_recovery_mode,',
    )

# Add recovery_mode to fieldnames.
if '"recovery_mode",' not in text:
    text = text.replace(
        '"frame_path", "action", "buttons", "trap",',
        '"frame_path", "action", "buttons", "trap", "recovery_mode",',
    )

p.write_text(text)
print("Patched wall recovery recorder with half-wall rail traps and recovery_mode.")
