from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_action_prior_after_compute_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """        reward = self._compute_reward(action_name, state)
"""

new = """        reward = self._compute_reward(action_name, state)

        # -------------------------------------------------
        # Supervised teacher / action-prior advice reward
        # -------------------------------------------------
        # This is advice-only. It does not override PPO.
        try:
            current_state_for_teacher = self.game.get_state()
            if current_state_for_teacher is not None:
                teacher_frame = current_state_for_teacher.screen_buffer
                teacher_reward, teacher_info = self.action_prior_advice_reward(
                    teacher_frame,
                    action_name,
                )
                reward += teacher_reward

                if hasattr(self, "reward_manager") and self.reward_manager is not None:
                    self.reward_manager.add("viz_action_prior_advice", teacher_reward)

                self.last_action_prior_info = teacher_info
        except Exception as e:
            self._viz_action_prior_error_count = getattr(self, "_viz_action_prior_error_count", 0) + 1
            if self._viz_action_prior_error_count <= 3:
                print(f"[viz_action_prior] reward hook failed: {e}")
"""

if "viz_action_prior_advice" in text and "backup_before_action_prior_after_compute_reward" not in text:
    print("A viz_action_prior_advice hook already exists somewhere. Still checking exact insertion...")

if old not in text:
    raise SystemExit("Could not find reward = self._compute_reward(action_name, state).")

if "teacher_reward, teacher_info = self.action_prior_advice_reward" in text:
    print("Teacher reward hook already exists. No changes made.")
else:
    text = text.replace(old, new, 1)
    p.write_text(text)
    print("Inserted action-prior reward directly after _compute_reward.")
