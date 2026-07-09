from pathlib import Path

doom_path = Path("env/doom_env.py")
brain_path = Path("rewards/doom_brain_reward.py")

doom = doom_path.read_text()
brain = brain_path.read_text()

doom_backup = Path("env/doom_env.py.backup_before_reduce_positive_3x")
brain_backup = Path("rewards/doom_brain_reward.py.backup_before_reduce_positive_3x")

doom_backup.write_text(doom)
brain_backup.write_text(brain)

print(f"Backup saved to {doom_backup}")
print(f"Backup saved to {brain_backup}")

# ---------------------------------------------------------
# 1. Change DoomEnv positive multiplier comments/logic 4x -> 3x
# ---------------------------------------------------------

doom = doom.replace(
    "Quadruple positive rewards.",
    "Triple positive rewards."
)

doom = doom.replace(
    "Quadruple positive rewards",
    "Triple positive rewards"
)

doom = doom.replace(
    "+0.35 -> +1.40",
    "+0.35 -> +1.05"
)

doom = doom.replace(
    "+3.00 -> +12.00",
    "+3.00 -> +9.00"
)

doom = doom.replace(
    "+15.00 -> +60.00",
    "+15.00 -> +45.00"
)

doom = doom.replace(
    "boosted = value * 4.0",
    "boosted = value * 3.0"
)

# ---------------------------------------------------------
# 2. Reduce specific manually boosted rewards from +60 to +45
# ---------------------------------------------------------

doom = doom.replace("60.0", "45.0")

# ---------------------------------------------------------
# 3. Change doom_brain_reward multiplier 4x -> 3x
# ---------------------------------------------------------

brain = brain.replace(
    "POSITIVE_REWARD_TEST_MULTIPLIER = 4.0",
    "POSITIVE_REWARD_TEST_MULTIPLIER = 3.0"
)

# ---------------------------------------------------------
# 4. Write files
# ---------------------------------------------------------

doom_path.write_text(doom)
brain_path.write_text(brain)

print("Reduced positive reward test multiplier from 4x to 3x.")
print("Specific +60 rewards were reduced to +45.")
