from pathlib import Path

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_action_prior_debug")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """        if confidence >= self.action_prior_min_confidence:
            if advised_action == action:
                reward += self.action_prior_reward_scale
            else:
                reward -= self.action_prior_reward_scale * 0.25

        return float(reward), result
"""

new = """        if confidence >= self.action_prior_min_confidence:
            if advised_action == action:
                reward += self.action_prior_reward_scale
            else:
                reward -= self.action_prior_reward_scale * 0.25

        step = getattr(self, "step_count", getattr(self, "current_step", 0))
        if step % 100 == 0:
            print(
                f"[action_prior_reward] step={step} "
                f"advised={advised_action} action={action} "
                f"conf={confidence:.2f} reward={reward:.4f}"
            )

        return float(reward), result
"""

if old not in text:
    raise SystemExit("Could not find action_prior_advice_reward block to patch.")

text = text.replace(old, new, 1)
p.write_text(text)
print("Added action-prior debug logging.")
