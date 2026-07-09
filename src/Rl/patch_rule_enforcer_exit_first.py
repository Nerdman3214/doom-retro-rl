from pathlib import Path

p = Path("director/doom_rule_enforcer.py")
text = p.read_text()

backup = Path("director/doom_rule_enforcer.py.backup_before_exit_first")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Stronger target progress, weaker generic safe movement.
text = text.replace(
    "reward += min(2.0, improvement / 32.0)",
    "reward += min(8.0, improvement / 16.0)"
)

text = text.replace(
    "reward += 1.0\n            events.append(\"good_safe_navigation_movement\")",
    "reward += 0.10\n            events.append(\"good_safe_navigation_movement\")"
)

# Make wrong-way stronger.
text = text.replace(
    "reward -= 2.0\n            events.append(\"navigation_wrong_way\")",
    "reward -= 8.0\n            events.append(\"navigation_wrong_way\")"
)

# Make combat awareness less dominant for now.
text = text.replace(
    "reward += 0.5\n            events.append(\"enemy_awareness\")",
    "reward += 0.05\n            events.append(\"enemy_awareness\")"
)

text = text.replace(
    "reward += 1.0\n            events.append(\"enemy_centered\")",
    "reward += 0.10\n            events.append(\"enemy_centered\")"
)

# Keep kills important but not bigger than exit.
text = text.replace(
    "reward += 20.0 * kill_delta",
    "reward += 10.0 * kill_delta"
)

# Level complete must dominate.
text = text.replace(
    "reward += 100.0\n            events.append(\"level_complete\")",
    "reward += 500.0\n            events.append(\"level_complete\")"
)

p.write_text(text)
print("Patched DoomRuleEnforcer for exit-first priority.")
