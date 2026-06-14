from pathlib import Path

p = Path("director/doom_rule_enforcer.py")
text = p.read_text()

backup = Path("director/doom_rule_enforcer.py.backup_before_old_negative_scale")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Return negatives to old-ish scale.
text = text.replace(
    "return -abs(float(amount)) * 6.0",
    "return -abs(float(amount)) * 1.0",
)

text = text.replace(
    "return -abs(float(amount)) * 3.0",
    "return -abs(float(amount)) * 1.0",
)

# Clip less harshly negative than the 6x test.
text = text.replace(
    "return max(-60.0, min(120.0, float(reward)))",
    "return max(-30.0, min(120.0, float(reward)))",
)

text = text.replace(
    "return max(-120.0, min(120.0, float(reward)))",
    "return max(-30.0, min(120.0, float(reward)))",
)

# Keep wasted shooting bad, but learnable.
text = text.replace("self._neg(10.0)", "self._neg(4.0)")
text = text.replace("self._neg(8.0)", "self._neg(4.0)")
text = text.replace("self._neg(6.0)", "self._neg(3.0)")

# Make useful shooting clearly positive.
text = text.replace(
    'reward += 18.0\n                events.append("good_shoot_centered_enemy")',
    'reward += 12.0\n                events.append("good_shoot_centered_enemy")',
)

text = text.replace(
    'reward += 8.0\n                events.append("good_shoot_centered_enemy")',
    'reward += 12.0\n                events.append("good_shoot_centered_enemy")',
)

text = text.replace(
    'reward += 6.0\n                events.append("okay_shoot_visible_enemy")',
    'reward += 4.0\n                events.append("okay_shoot_visible_enemy")',
)

# Make useful use logic.
if "good_use_near_door" in text:
    text = text.replace(
        'reward += 8.0\n                events.append("good_use_near_door")',
        'reward += 10.0\n                events.append("good_use_near_door")',
    )
    text = text.replace(
        'reward += self._neg(4.0)\n                events.append("bad_use_spam")',
        'reward += self._neg(2.0)\n                events.append("bad_use_spam")',
    )

p.write_text(text)
print("Restored old-ish negative scale and added learnable shoot/use rewards.")
