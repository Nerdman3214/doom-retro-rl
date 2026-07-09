from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_recovery_action_prior_safe")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Ensure imports exist before the recovery class.
# ------------------------------------------------------------
extra_imports = """
# Recovery prior imports
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path as _RecoveryPath
"""

if "# Recovery prior imports" not in text:
    # Insert after initial imports, safely near the top.
    text = extra_imports + "\n" + text

# ------------------------------------------------------------
# 2. Add module-level advisor and helper functions before env class.
# ------------------------------------------------------------
if "class RecoveryActionPriorAdvisor" not in text:
    recovery_block = r'''
class RecoveryActionPriorAdvisor:
    RECOVERY_ALLOWED = {
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
    }

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = _RecoveryPath(checkpoint_path)
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

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3:
            return None

        if arr.shape[-1] > 3:
            arr = arr[..., :3]

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
        ranked = torch.argsort(probs, descending=True).detach().cpu().tolist()

        for idx in ranked:
            if idx < 0 or idx >= len(self.action_names):
                continue

            name = self.action_names[idx]
            conf = float(probs[idx].detach().cpu().item())

            # Important: never use shoot/use as recovery advice.
            if name in self.RECOVERY_ALLOWED:
                return name, conf

        return None, 0.0


def recovery_prior_is_recovery_state(env):
    wall_contact_steps = int(getattr(env, "wall_contact_steps", 0) or 0)
    corner_trap_steps = int(getattr(env, "corner_trap_steps", 0) or 0)
    stuck_steps = int(getattr(env, "stuck_steps", 0) or 0)
    no_progress_steps = int(getattr(env, "no_progress_steps", 0) or 0)
    recovery_mode = bool(getattr(env, "recovery_mode", False))

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


def recovery_action_prior_reward(env, frame, action_name):
    if not bool(getattr(env, "use_recovery_action_prior", False)):
        return 0.0

    if not recovery_prior_is_recovery_state(env):
        return 0.0

    advisor = getattr(env, "recovery_action_prior", None)
    if advisor is None or not getattr(advisor, "ready", False):
        return 0.0

    advised_action, confidence = advisor.advise(frame)

    env.recovery_prior_call_count = int(getattr(env, "recovery_prior_call_count", 0)) + 1

    if advised_action is None:
        return 0.0

    min_conf = float(getattr(env, "recovery_prior_min_confidence", 0.20))

    if confidence < min_conf:
        reward = 0.0
    elif str(action_name) == str(advised_action):
        reward = float(getattr(env, "recovery_prior_reward_scale", 0.04))
    else:
        reward = float(getattr(env, "recovery_prior_mismatch_penalty", -0.004))

    every = int(getattr(env, "recovery_prior_debug_every", 100))
    if every > 0 and env.recovery_prior_call_count % every == 0:
        print(
            f"[recovery_prior_reward] call={env.recovery_prior_call_count} "
            f"advised={advised_action} action={action_name} "
            f"conf={confidence:.2f} reward={reward:.4f} "
            f"wall={getattr(env, 'wall_contact_steps', 0)} "
            f"stuck={getattr(env, 'stuck_steps', 0)} "
            f"noprog={getattr(env, 'no_progress_steps', 0)}"
        )

    return float(reward)
'''

    m = re.search(r"\nclass\s+(VizDoomEnv|DoomEnv)\b", text)
    if not m:
        raise SystemExit("Could not find VizDoomEnv/DoomEnv class header.")

    text = text[:m.start()] + "\n" + recovery_block + "\n" + text[m.start():]
    print("Inserted recovery advisor/helper block.")
else:
    print("Recovery advisor already exists.")

# ------------------------------------------------------------
# 3. Add env __init__ fields.
# ------------------------------------------------------------
if "self.recovery_action_prior = RecoveryActionPriorAdvisor" not in text:
    init_fields = '''
        # Recovery action prior: used only when stuck/wall-blocked.
        self.use_recovery_action_prior = True
        self.recovery_prior_min_confidence = 0.20
        self.recovery_prior_reward_scale = 0.04
        self.recovery_prior_mismatch_penalty = -0.004
        self.recovery_prior_debug_every = 100
        self.recovery_prior_call_count = 0
        self.recovery_action_prior = RecoveryActionPriorAdvisor(
            _RecoveryPath(__file__).resolve().parents[1] / "checkpoints" / "recovery_action_prior.pt"
        )
'''

    m = re.search(r"\n    def __init__\(.*?\):\n", text)
    if not m:
        raise SystemExit("Could not find env __init__.")

    text = text[:m.end()] + init_fields + text[m.end():]
    print("Inserted recovery prior __init__ fields.")
else:
    print("Recovery prior __init__ fields already exist.")

# ------------------------------------------------------------
# 4. Hook into reward computation with exact indentation.
# ------------------------------------------------------------
if "recovery_action_prior_reward(self, recovery_frame, action_name)" not in text:
    pattern = re.compile(r"^(?P<indent>\s*)reward\s*=\s*self\._compute_reward\(action_name,\s*state\)\s*$", re.MULTILINE)
    m = pattern.search(text)

    if not m:
        raise SystemExit("Could not find exact reward = self._compute_reward(action_name, state) line.")

    indent = m.group("indent")
    hook = (
        "\n"
        f"{indent}# Recovery-prior reward only activates during stuck/wall-blocked states.\n"
        f"{indent}try:\n"
        f"{indent}    recovery_frame = None\n"
        f"{indent}    if 'post_game_state' in locals() and post_game_state is not None:\n"
        f"{indent}        recovery_frame = getattr(post_game_state, 'screen_buffer', None)\n"
        f"{indent}    if recovery_frame is None and 'state' in locals() and state is not None:\n"
        f"{indent}        recovery_frame = getattr(state, 'screen_buffer', None)\n"
        f"{indent}    reward += recovery_action_prior_reward(self, recovery_frame, action_name)\n"
        f"{indent}except Exception as e:\n"
        f"{indent}    if int(getattr(self, 'recovery_prior_call_count', 0)) % 500 == 0:\n"
        f"{indent}        print(f'[recovery_prior_reward] skipped due error: {{e}}')\n"
    )

    insert_at = m.end()
    text = text[:insert_at] + hook + text[insert_at:]
    print("Inserted recovery reward hook.")
else:
    print("Recovery reward hook already exists.")

p.write_text(text)
print("Safe patch complete.")
