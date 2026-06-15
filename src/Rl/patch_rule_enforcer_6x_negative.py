from pathlib import Path

p = Path("director/doom_rule_enforcer.py")
text = p.read_text()

backup = Path("director/doom_rule_enforcer.py.backup_before_6x_negative")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add helper methods if missing.
if "def _neg(self, amount):" not in text:
    marker = "    def _dist(self, ax, ay, bx, by):\n"
    helper = '''    def _neg(self, amount):
        """
        Test mode:
        make negative rewards much stronger.
        """
        return -abs(float(amount)) * 6.0

    def _pos(self, amount):
        """
        Keep positive rewards normal for this test.
        """
        return abs(float(amount))

'''
    text = text.replace(marker, helper + marker, 1)

# Replace key negative rewards with 6x helper.
replacements = {
    "reward -= 15.0": "reward += self._neg(15.0)",
    "reward -= 8.0": "reward += self._neg(8.0)",
    "reward -= 6.0": "reward += self._neg(6.0)",
    "reward -= 4.0": "reward += self._neg(4.0)",
    "reward -= 2.0": "reward += self._neg(2.0)",
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Make useful shooting clear.
text = text.replace(
    '''if action_name == "shoot":
            if enemy_visible and enemy_centered and ammo > 0:
                reward += 6.0
                events.append("good_shoot_centered_enemy")
            elif enemy_visible and ammo > 0:
                reward += 2.0
                events.append("okay_shoot_visible_enemy")
            else:
                reward += self._neg(6.0)
                events.append("bad_blind_shoot")''',
    '''if action_name == "shoot":
            if enemy_visible and enemy_centered and ammo > 0:
                reward += 8.0
                events.append("good_shoot_centered_enemy")
            elif enemy_visible and ammo > 0:
                reward += 3.0
                events.append("okay_shoot_visible_enemy")
            elif ammo <= 0:
                reward += self._neg(8.0)
                events.append("bad_shoot_no_ammo")
            else:
                reward += self._neg(10.0)
                events.append("bad_blind_or_wasted_shot")'''
)

# If the exact block did not match, add a second-level wasted-shot penalty near basic fighting rule.
if "bad_blind_or_wasted_shot" not in text:
    marker = '''        if kill_delta > 0:
            reward += 10.0 * kill_delta
            events.append("enemy_kill")
'''
    insert = '''        if action_name == "shoot" and not enemy_visible:
            reward += self._neg(10.0)
            events.append("bad_blind_or_wasted_shot")

        if action_name == "shoot" and enemy_visible and not enemy_centered:
            reward += self._neg(3.0)
            events.append("bad_poorly_aimed_shot")

'''
    if marker in text:
        text = text.replace(marker, insert + marker, 1)

# Add reward clipping to avoid exploding PPO value loss too hard.
if "def _clip_total_reward" not in text:
    marker = "    def preferred_action(self, game_state, target=None):\n"
    clipper = '''    def _clip_total_reward(self, reward):
        """
        Keep the 6x-negative test strong but bounded enough for PPO.
        """
        return max(-120.0, min(120.0, float(reward)))

'''
    text = text.replace(marker, clipper + marker, 1)

text = text.replace(
    '''        return {
            "reward": float(total),''',
    '''        total = self._clip_total_reward(total)

        return {
            "reward": float(total),'''
)

p.write_text(text)
print("Patched DoomRuleEnforcer with 6x negative rewards and strong wasted-shot penalties.")
