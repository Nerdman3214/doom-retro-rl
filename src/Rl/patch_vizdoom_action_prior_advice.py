from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_action_prior_advice")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add import.
if "from imitation.action_prior_advisor import ActionPriorAdvisor" not in text:
    marker = "import torch\n"
    if marker in text:
        text = text.replace(
            marker,
            marker + "from imitation.action_prior_advisor import ActionPriorAdvisor\n",
            1,
        )
    else:
        text = "from imitation.action_prior_advisor import ActionPriorAdvisor\n" + text

# Add advisor init after RewardManager init.
init_marker = "        self.reward_manager = RewardManager()\n"

init_block = """        self.use_action_prior_advice = True
        self.action_prior_reward_scale = 0.02
        self.action_prior_min_confidence = 0.40

        action_prior_path = str(ROOT / "checkpoints" / "action_prior_from_teacher_balanced.pt")

        try:
            self.action_prior_advisor = ActionPriorAdvisor(
                checkpoint_path=action_prior_path,
                action_names=[
                    "move_forward",
                    "turn_left",
                    "turn_right",
                    "strafe_left",
                    "strafe_right",
                    "move_backward",
                    "shoot",
                    "use",
                ],
            )
        except Exception as e:
            print(f"[viz_action_prior] disabled: {e}")
            self.action_prior_advisor = None

"""

if "self.action_prior_reward_scale = 0.02" not in text:
    if init_marker not in text:
        raise SystemExit("Could not find self.reward_manager = RewardManager() marker.")
    text = text.replace(init_marker, init_marker + init_block, 1)

# Add helper method.
method = '''
    def action_prior_advice_reward(self, frame, action_name):
        """
        Advice-only reward from the human/teacher action-prior model.

        This does not override PPO. It gives a small reward when PPO agrees
        with the supervised teacher.
        """
        if not getattr(self, "use_action_prior_advice", False):
            return 0.0, None

        advisor = getattr(self, "action_prior_advisor", None)
        if advisor is None:
            return 0.0, None

        result = advisor.predict(frame)

        if not result.get("enabled", False):
            return 0.0, result

        advised_action = result.get("action_name")
        confidence = float(result.get("confidence", 0.0))

        reward = 0.0

        if confidence >= self.action_prior_min_confidence:
            if advised_action == action_name:
                reward += self.action_prior_reward_scale
            else:
                reward -= self.action_prior_reward_scale * 0.10

        step = getattr(self, "step_count", getattr(self, "episode_step", 0))
        if step % 100 == 0:
            print(
                f"[viz_action_prior_reward] step={step} "
                f"advised={advised_action} action={action_name} "
                f"conf={confidence:.2f} reward={reward:.4f}"
            )

        return float(reward), result

'''

if "def action_prior_advice_reward(self, frame, action_name):" not in text:
    insert_before = "    def route_progress_reward"
    if insert_before not in text:
        raise SystemExit("Could not find route_progress_reward insertion point.")
    text = text.replace(insert_before, method + "\n" + insert_before, 1)

p.write_text(text)
print("Patched ViZDoom env with action-prior advisor.")
