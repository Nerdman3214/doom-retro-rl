from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_action_prior_debug_counter")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = '''        step = getattr(self, "step_count", getattr(self, "episode_step", 0))
        if step % 100 == 0:
            print(
                f"[viz_action_prior_reward] step={step} "
                f"advised={advised_action} action={action_name} "
                f"conf={confidence:.2f} reward={reward:.4f}"
            )
'''

new = '''        self._viz_action_prior_debug_count = getattr(self, "_viz_action_prior_debug_count", 0) + 1
        if self._viz_action_prior_debug_count % 100 == 0:
            step = getattr(self, "step_count", getattr(self, "episode_step", self._viz_action_prior_debug_count))
            print(
                f"[viz_action_prior_reward] call={self._viz_action_prior_debug_count} "
                f"step={step} advised={advised_action} action={action_name} "
                f"conf={confidence:.2f} reward={reward:.4f}"
            )
'''

if old not in text:
    raise SystemExit("Could not find old viz action-prior debug block.")

text = text.replace(old, new, 1)
p.write_text(text)
print("Patched ViZDoom action-prior debug counter.")
