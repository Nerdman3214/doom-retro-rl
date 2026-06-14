from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_insert_action_prior_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """            reward += float(self.game.get_total_reward())
"""

new = """            reward += float(self.game.get_total_reward())

            # -------------------------------------------------
            # Supervised teacher / action-prior advice reward
            # -------------------------------------------------
            # This is advice-only. It does not override PPO.
            # It gives a small reward when PPO agrees with the
            # human-trained action-prior model.
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
                step = getattr(self, "step_count", getattr(self, "episode_step", 0))
                if step % 250 == 0:
                    print(f"[viz_action_prior] reward hook failed: {e}")
"""

if old not in text:
    raise SystemExit("Could not find ViZDoom game reward line to patch.")

if "viz_action_prior_advice" in text:
    print("Action-prior reward hook already appears to be inserted. No changes made.")
else:
    text = text.replace(old, new, 1)
    p.write_text(text)
    print("Inserted ViZDoom action-prior reward hook.")
