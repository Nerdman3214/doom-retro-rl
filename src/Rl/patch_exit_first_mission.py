from pathlib import Path

p = Path("navigation/mission_plan.py")
text = p.read_text()

backup = Path("navigation/mission_plan.py.backup_before_exit_first")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Replace route objective rewards with tiny guidance.
# The agent should not think "corridor reached" is success.
for old, new in [
    ("reward=0.15,", "reward=0.00,"),
    ("reward=0.30,", "reward=0.00,"),
    ("reward=0.40,", "reward=0.00,"),
    ("reward=0.80,", "reward=0.00,"),
    ("reward=1.00,", "reward=0.00,"),
    ("reward=1.50,", "reward=0.00,"),
]:
    text = text.replace(old, new)

# Secrets should wait until the agent can finish the level.
text = text.replace("reward=0.60,", "reward=0.00,")

# Make actual exit reward much stronger.
text = text.replace(
    'name="level_exit",\n            kind="exit",\n            x=-400.0,\n            y=1296.0,\n            radius=128.0,\n            reward=5.00,',
    'name="level_exit",\n            kind="exit",\n            x=-400.0,\n            y=1296.0,\n            radius=160.0,\n            reward=100.00,'
)

# Make dense exit progress stronger than route bubble progress.
text = text.replace(
    "result[\"reward\"] += min(0.12, exit_improvement * 0.002)",
    "result[\"reward\"] += min(2.00, exit_improvement * 0.020)"
)

# Make current target progress smaller so it does not farm corridor bubbles.
text = text.replace(
    "shaped = min(0.25, improvement * 0.006)",
    "shaped = min(0.05, improvement * 0.001)"
)

p.write_text(text)
print("Patched mission_plan.py for exit-first training.")
