from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_recovery_action_prior")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Ensure imports exist.
# ------------------------------------------------------------
if "from collections import deque" not in text and "import deque" not in text:
    text = text.replace("import random\n", "import random\nfrom collections import deque\n", 1)

# ------------------------------------------------------------
# 2. Add RecoveryActionPriorAdvisor class.
# ------------------------------------------------------------
if "class RecoveryActionPriorAdvisor" not in text:
    insert_class = r'''
class RecoveryActionPriorAdvisor:
    """
    Small CNN advisor trained on wall/half-wall recovery recordings.

    This is only used during recovery-like states. It should advise movement
    actions such as turn/strafe/move, not shoot/use.
    """
    RECOVERY_ALLOWED = {
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
    }

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.action_names = []
        self.ready = False

        if not self.checkpoint_path.exists():
            print(f"[recovery_prior] missing {self.checkpoint_path}")
            return

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.action_names = list(ckpt.get("action_names", []))
            num_actions = int(ckpt.get("num_actions", len(self.action_names) or 8))

            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions),
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.ready = True

            print(
                f"[recovery_prior] loaded {self.checkpoint_path} "
                f"num_actions={num_actions} device={self.device}"
            )
        except Exception as e:
            print(f"[recovery_prior] failed to load {self.checkpoint_path}: {e}")
            self.ready = False

    def preprocess(self, frame):
        if frame is None:
            return None

        arr = np.asarray(frame)

        # CHW -> HWC if needed.
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Ensure uint8-ish RGB.
        arr = arr.astype(np.uint8)

        import cv2
        arr = cv2.resize(arr, (84, 84), interpolation=cv2.INTER_AREA)
        arr = arr.astype(np.float32) / 255.0

        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def advise(self, frame):
        if not self.ready or self.model is None:
            return None, 0.0

        x = self.preprocess(frame)
        if x is None:
            return None, 0.0

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        # Prefer the best recovery-allowed action. Ignore shoot/use.
        ranked = torch.argsort(probs, descending=True).detach().cpu().tolist()
        for idx in ranked:
            if idx < 0 or idx >= len(self.action_names):
                continue
            name = self.action_names[idx]
            conf = float(probs[idx].detach().cpu().item())
            if name in self.RECOVERY_ALLOWED:
                return name, conf

        return None, 0.0
'''
    # Put it before the env class.
    m = re.search(r"\nclass\s+\w*DoomEnv\b|\nclass\s+VizDoomEnv\b", text)
    if not m:
        raise SystemExit("Could not find env class. Paste class definition header.")
    text = text[:m.start()] + insert_class + "\n" + text[m.start():]
    print("Inserted RecoveryActionPriorAdvisor class.")
else:
    print("RecoveryActionPriorAdvisor already present.")

# ------------------------------------------------------------
# 3. Add init fields after normal action prior setup or in __init__.
# ------------------------------------------------------------
if "self.recovery_action_prior" not in text:
    # Insert near action_prior if possible.
    init_insert = '''
        # Recovery action prior: only used when the agent appears stuck/wall-blocked.
        self.use_recovery_action_prior = True
        self.recovery_prior_min_confidence = 0.20
        self.recovery_prior_reward_scale = 0.04
        self.recovery_prior_mismatch_penalty = -0.004
        self.recovery_prior_debug_every = 100
        self.recovery_prior_call_count = 0
        self.recovery_action_prior = RecoveryActionPriorAdvisor(
            ROOT / "checkpoints" / "recovery_action_prior.pt"
        )
'''
    # Try inserting after action_prior initialization.
    patterns = [
        "self.action_prior =",
        "self.action_prior_advisor =",
        "self.use_action_prior",
    ]

    inserted = False
    for pat in patterns:
        idx = text.find(pat)
        if idx != -1:
            # Insert after the surrounding line block.
            line_end = text.find("\n", idx)
            text = text[:line_end + 1] + init_insert + text[line_end + 1:]
            inserted = True
            print(f"Inserted recovery prior init after {pat}.")
            break

    if not inserted:
        # Fallback: insert after first __init__ line.
        m = re.search(r"def __init__\(.*?\):\n", text)
        if not m:
            raise SystemExit("Could not find __init__ to insert recovery prior fields.")
        text = text[:m.end()] + init_insert + text[m.end():]
        print("Inserted recovery prior init after __init__ header.")
else:
    print("recovery_action_prior init already present.")

# ------------------------------------------------------------
# 4. Add helper methods inside env class before step/reset helpers.
# ------------------------------------------------------------
if "def is_recovery_state" not in text:
    helper_methods = r'''
    def is_recovery_state(self, game_state=None):
        """
        Heuristic stuck detector for recovery prior.

        We keep this conservative:
        - wall/contact counters if they exist
        - recent low movement/progress counters if they exist
        - long repeated route stall if present
        """
        wall_contact_steps = int(getattr(self, "wall_contact_steps", 0) or 0)
        corner_trap_steps = int(getattr(self, "corner_trap_steps", 0) or 0)
        stuck_steps = int(getattr(self, "stuck_steps", 0) or 0)
        no_progress_steps = int(getattr(self, "no_progress_steps", 0) or 0)
        recovery_mode = bool(getattr(self, "recovery_mode", False))

        if recovery_mode:
            return True

        if wall_contact_steps >= 8:
            return True

        if corner_trap_steps >= 6:
            return True

        if stuck_steps >= 10:
            return True

        if no_progress_steps >= 18:
            return True

        return False

    def recovery_action_prior_reward(self, frame, action_name, game_state=None):
        """
        Reward matching the recovery prior only during recovery states.
        Does not reward shoot/use; the advisor filters those out.
        """
        if not bool(getattr(self, "use_recovery_action_prior", False)):
            return 0.0

        if not self.is_recovery_state(game_state):
            return 0.0

        advisor = getattr(self, "recovery_action_prior", None)
        if advisor is None or not getattr(advisor, "ready", False):
            return 0.0

        advised_action, confidence = advisor.advise(frame)
        self.recovery_prior_call_count = int(getattr(self, "recovery_prior_call_count", 0)) + 1

        if advised_action is None:
            return 0.0

        min_conf = float(getattr(self, "recovery_prior_min_confidence", 0.20))
        if confidence < min_conf:
            reward = 0.0
        elif str(action_name) == str(advised_action):
            reward = float(getattr(self, "recovery_prior_reward_scale", 0.04))
        else:
            reward = float(getattr(self, "recovery_prior_mismatch_penalty", -0.004))

        every = int(getattr(self, "recovery_prior_debug_every", 100))
        if every > 0 and self.recovery_prior_call_count % every == 0:
            print(
                f"[recovery_prior_reward] call={self.recovery_prior_call_count} "
                f"advised={advised_action} action={action_name} "
                f"conf={confidence:.2f} reward={reward:.4f} "
                f"wall={getattr(self, 'wall_contact_steps', 0)} "
                f"stuck={getattr(self, 'stuck_steps', 0)} "
                f"noprog={getattr(self, 'no_progress_steps', 0)}"
            )

        return float(reward)
'''
    # Insert before action_prior_advice_reward if present, otherwise before reset.
    target = "def action_prior_advice_reward"
    idx = text.find(target)
    if idx == -1:
        target = "def reset"
        idx = text.find(target)
    if idx == -1:
        raise SystemExit("Could not find insertion point for recovery helper methods.")

    # Need class indentation. This text already starts with 4 spaces.
    text = text[:idx] + helper_methods + "\n" + text[idx:]
    print("Inserted recovery helper methods.")
else:
    print("Recovery helper methods already present.")

# ------------------------------------------------------------
# 5. Hook reward after normal reward computation.
# ------------------------------------------------------------
if "recovery_action_prior_reward(" not in text[text.find("def step"):]:
    # Common reward line.
    candidates = [
        "reward = self._compute_reward(action_name, state)",
        "reward = self._compute_reward(action_name, post_game_state)",
        "reward = self._compute_reward(action_name",
    ]

    hook = '''
        # Recovery-prior reward only activates when the agent appears stuck/wall-blocked.
        try:
            recovery_frame = None
            if "post_game_state" in locals() and post_game_state is not None:
                recovery_frame = getattr(post_game_state, "screen_buffer", None)
            if recovery_frame is None and "state" in locals() and state is not None:
                recovery_frame = getattr(state, "screen_buffer", None)
            reward += self.recovery_action_prior_reward(
                recovery_frame,
                action_name,
                locals().get("post_game_state", locals().get("state", None)),
            )
        except Exception as e:
            if int(getattr(self, "recovery_prior_call_count", 0)) % 500 == 0:
                print(f"[recovery_prior_reward] skipped due error: {e}")
'''

    inserted = False
    for cand in candidates:
        idx = text.find(cand)
        if idx != -1:
            line_end = text.find("\n", idx)
            text = text[:line_end + 1] + hook + text[line_end + 1:]
            inserted = True
            print(f"Inserted recovery reward hook after: {cand}")
            break

    if not inserted:
        raise SystemExit("Could not find reward computation line. Paste step() reward section.")
else:
    print("Recovery reward hook already present.")

p.write_text(text)
print("Patch complete.")
