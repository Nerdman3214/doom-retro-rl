from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_behavior_judge_reward")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Add imports
# ------------------------------------------------------------
if "from models.behavior_judge_model import BehaviorJudgeCNN" not in text:
    marker = "from imitation.action_prior_advisor import ActionPriorAdvisor\n"
    if marker in text:
        text = text.replace(
            marker,
            marker + "from models.behavior_judge_model import BehaviorJudgeCNN\n",
            1,
        )
    else:
        text = "from models.behavior_judge_model import BehaviorJudgeCNN\n" + text

if "from collections import deque" not in text:
    text = text.replace("import os\n", "import os\nfrom collections import deque\n", 1)

if "import numpy as np" not in text:
    text = text.replace("import torch\n", "import torch\nimport numpy as np\n", 1)

# ------------------------------------------------------------
# 2. Add init block after action-prior setup
# ------------------------------------------------------------
marker = "        self.action_prior_min_confidence = 0.40\n"

init_block = """        # -------------------------------------------------
        # RLHF-style behavior judge reward
        # -------------------------------------------------
        self.use_behavior_judge_reward = True
        self.behavior_judge_good_threshold = 0.70
        self.behavior_judge_bad_threshold = 0.70
        self.behavior_judge_good_reward = 0.02
        self.behavior_judge_bad_penalty = -0.05
        self.behavior_judge_eval_interval = 8
        self.behavior_judge_frame_history = deque(maxlen=3)
        self._behavior_judge_debug_count = 0

        behavior_judge_path = ROOT / "checkpoints" / "behavior_judge.pt"
        self.behavior_judge = None
        self.behavior_judge_device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if behavior_judge_path.exists():
                ckpt = torch.load(str(behavior_judge_path), map_location=self.behavior_judge_device)
                self.behavior_judge = BehaviorJudgeCNN().to(self.behavior_judge_device)
                self.behavior_judge.load_state_dict(ckpt["model_state_dict"])
                self.behavior_judge.eval()
                print(
                    f"[behavior_judge] loaded {behavior_judge_path} "
                    f"device={self.behavior_judge_device} "
                    f"type={ckpt.get('model_type', 'unknown')}"
                )
            else:
                print(f"[behavior_judge] missing checkpoint: {behavior_judge_path}")
        except Exception as e:
            print(f"[behavior_judge] disabled: {e}")
            self.behavior_judge = None

"""

if "self.use_behavior_judge_reward = True" not in text:
    if marker not in text:
        raise SystemExit("Could not find action_prior_min_confidence marker.")
    text = text.replace(marker, marker + init_block, 1)

# ------------------------------------------------------------
# 3. Add helper methods before action_prior_advice_reward
# ------------------------------------------------------------
helper = '''
    def _behavior_judge_preprocess_frame(self, frame):
        """
        Convert ViZDoom screen buffer into [3, 84, 84] float tensor.
        """
        arr = frame

        if arr is None:
            return None

        arr = np.asarray(arr)

        # ViZDoom often gives [C, H, W].
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        # Grayscale -> RGB.
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # If RGBA or more channels, keep RGB.
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Resize using PIL because torchvision is broken in this environment.
        from PIL import Image

        img = Image.fromarray(arr.astype("uint8")).convert("RGB")
        img = img.resize((84, 84))

        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        return torch.tensor(arr, dtype=torch.float32)

    def behavior_judge_reward(self, frame):
        """
        RLHF-style reward from the 3-frame behavior judge.

        This does not control the agent. It only adds a small reward/penalty
        when the judge is confident.
        """
        if not getattr(self, "use_behavior_judge_reward", False):
            return 0.0, None

        judge = getattr(self, "behavior_judge", None)
        if judge is None:
            return 0.0, None

        tensor = self._behavior_judge_preprocess_frame(frame)
        if tensor is None:
            return 0.0, None

        self.behavior_judge_frame_history.append(tensor)

        if len(self.behavior_judge_frame_history) < 3:
            return 0.0, None

        step = getattr(self, "step_count", getattr(self, "episode_step", 0))
        interval = getattr(self, "behavior_judge_eval_interval", 8)

        if interval > 1 and step % interval != 0:
            return 0.0, None

        clip_tensor = torch.cat(list(self.behavior_judge_frame_history), dim=0)
        clip_tensor = clip_tensor.unsqueeze(0).to(self.behavior_judge_device)

        with torch.no_grad():
            logits = judge(clip_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        bad_prob = float(probs[0].item())
        good_prob = float(probs[1].item())

        reward = 0.0

        if bad_prob >= self.behavior_judge_bad_threshold:
            reward += self.behavior_judge_bad_penalty
        elif good_prob >= self.behavior_judge_good_threshold:
            reward += self.behavior_judge_good_reward

        info = {
            "bad_prob": bad_prob,
            "good_prob": good_prob,
            "reward": reward,
        }

        self._behavior_judge_debug_count = getattr(self, "_behavior_judge_debug_count", 0) + 1
        if self._behavior_judge_debug_count % 25 == 0:
            print(
                f"[behavior_judge_reward] call={self._behavior_judge_debug_count} "
                f"step={step} bad={bad_prob:.3f} good={good_prob:.3f} reward={reward:.4f}"
            )

        return float(reward), info

'''

if "def behavior_judge_reward(self, frame):" not in text:
    insert_before = "    def action_prior_advice_reward"
    if insert_before not in text:
        raise SystemExit("Could not find action_prior_advice_reward insertion point.")
    text = text.replace(insert_before, helper + "\n" + insert_before, 1)

# ------------------------------------------------------------
# 4. Insert reward hook after _compute_reward main path
# ------------------------------------------------------------
old = """        reward = self._compute_reward(action_name, state)
"""

new = """        reward = self._compute_reward(action_name, state)

        # -------------------------------------------------
        # RLHF-style behavior judge reward
        # -------------------------------------------------
        try:
            current_state_for_judge = self.game.get_state()
            if current_state_for_judge is not None:
                judge_frame = current_state_for_judge.screen_buffer
                judge_reward, judge_info = self.behavior_judge_reward(judge_frame)
                reward += judge_reward

                if hasattr(self, "reward_manager") and self.reward_manager is not None:
                    self.reward_manager.add("viz_behavior_judge", judge_reward)

                self.last_behavior_judge_info = judge_info
        except Exception as e:
            self._behavior_judge_error_count = getattr(self, "_behavior_judge_error_count", 0) + 1
            if self._behavior_judge_error_count <= 3:
                print(f"[behavior_judge] reward hook failed: {e}")
"""

if "viz_behavior_judge" not in text:
    if old not in text:
        raise SystemExit("Could not find reward = self._compute_reward(action_name, state).")
    text = text.replace(old, new, 1)

p.write_text(text)
print("Patched ViZDoom with behavior judge / RLHF reward.")
