from pathlib import Path
import re

ROOT = Path(".")
env_path = ROOT / "env" / "doom_env.py"
scene_path = ROOT / "vision" / "scene_predictor.py"
object_path = ROOT / "vision" / "object_predictor.py"

# ------------------------------------------------------------
# 1. Patch scene_predictor.py so predict() accepts view_mode
# ------------------------------------------------------------
scene = scene_path.read_text()

scene = scene.replace(
    "def predict(self, frame):",
    'def predict(self, frame, view_mode="center"):'
)

# Add safer BGRA handling if it is not already there.
scene = scene.replace(
    "# OpenCV uses BGR. PIL expects RGB.\n        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)",
    """# OpenCV uses BGR. PIL expects RGB.
        # view_mode is accepted for compatibility with DoomEnv.
        # The current trained classifier still uses its normal resize transform.
        if hasattr(frame, "shape") and len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)"""
)

scene_path.write_text(scene)


# ------------------------------------------------------------
# 2. Patch object_predictor.py so predict() accepts view_mode
#    and ObjectPredictor has frame_processor if local code uses it
# ------------------------------------------------------------
obj = object_path.read_text()

obj = obj.replace(
    "def predict(self, frame):",
    'def predict(self, frame, view_mode="wide"):'
)

# Add FrameProcessor import only if missing.
if "from observation.frame_processor import FrameProcessor" not in obj:
    obj = obj.replace(
        "from PIL import Image",
        "from PIL import Image\nfrom observation.frame_processor import FrameProcessor"
    )

# Add self.frame_processor inside __init__ only if missing.
if "self.frame_processor = FrameProcessor()" not in obj:
    obj = obj.replace(
        "self.thresholds = thresholds or OBJECT_THRESHOLDS",
        "self.thresholds = thresholds or OBJECT_THRESHOLDS\n        self.frame_processor = FrameProcessor()"
    )

object_path.write_text(obj)


# ------------------------------------------------------------
# 3. Patch doom_env.py runtime bugs
# ------------------------------------------------------------
env = env_path.read_text()

# Keep these helpers from hijacking PPO while debugging.
env = env.replace("self.enable_retrace_navigator = True", "self.enable_retrace_navigator = False")
env = env.replace("self.enable_goal_assist_action_override = True", "self.enable_goal_assist_action_override = False")

# Remove accidental bad return from inside step().
env = re.sub(
    r'\n\s*# If current weapon is already a strong general combat weapon,\n'
    r'\s*# do not swap just because an enemy is visible\.\n'
    r'\s*good_general_weapon = current_weapon in \["shotgun", "chaingun", "plasma"\]\n\n'
    r'\s*if good_general_weapon and enemy_visible:\n'
    r'\s*return None, f"keep_\{current_weapon\}_enemy_visible"\n',
    "\n",
    env
)

# Replace the invalid pre-action sensory placeholder block.
env = re.sub(
    r'        # -----------------------------------------------------\n'
    r'        # Pre-action sensory emergency controller\n'
    r'        # -----------------------------------------------------\n'
    r'        # This runs before wall bubble and before perform_action\(\)\.\n'
    r'        # If active, it becomes the single emergency recovery decision\.\n'
    r'        sensory_action = None\n\n'
    r'        if self\.enable_sensory_action_override:\n'
    r'            sensory_action = self\.sensory_emergency_action\(\.\.\.\)\n\n'
    r'            if sensory_action is not None:\n'
    r'                before = action\n'
    r'                action = sensory_action\n'
    r'                print\(f"\[sensory_pre_action\] \{before\} -> \{action\}"\)\n'
    r'                pre_game_state\["action"\] = action\n\n'
    r'            print\(\n'
    r'                f"\[sensory_pre_action\] \{before\} -> \{action\} "\n'
    r'                f"steps=\{self\.sensory_emergency_steps\} "\n'
    r'                f"L=\{pre_wall_info\.get\(\'left_ratio\', 0\.0\):\.2f\} "\n'
    r'                f"F=\{pre_wall_info\.get\(\'front_ratio\', 0\.0\):\.2f\} "\n'
    r'                f"R=\{pre_wall_info\.get\(\'right_ratio\', 0\.0\):\.2f\} "\n'
    r'                f"scene=\{pre_scene_label\} conf=\{pre_scene_confidence:\.2f\}"\n'
    r'            \)\n\n',
    '''        # -----------------------------------------------------
        # Pre-action sensory emergency controller
        # -----------------------------------------------------
        # This runs before wall bubble and before perform_action().
        # If active, it becomes the single emergency recovery decision.
        sensory_action = None

        if self.enable_sensory_action_override:
            sensory_action = self.sensory_emergency_action(
                action=action,
                scene_label=pre_scene_label,
                scene_confidence=pre_scene_confidence,
                wall_info=pre_wall_info,
                stuck_counter=self.stuck_counter,
                wall_contact_steps=self.wall_contact_steps,
                motion=None,
                distance_moved=None,
            )

            if sensory_action is not None and sensory_action in self.get_allowed_actions():
                before = action
                action = sensory_action
                pre_game_state["action"] = action

                print(
                    f"[sensory_pre_action] {before} -> {action} "
                    f"steps={self.sensory_emergency_steps} "
                    f"L={pre_wall_info.get('left_ratio', 0.0):.2f} "
                    f"F={pre_wall_info.get('front_ratio', 0.0):.2f} "
                    f"R={pre_wall_info.get('right_ratio', 0.0):.2f} "
                    f"scene={pre_scene_label} conf={pre_scene_confidence:.2f}"
                )

''',
    env,
    flags=re.DOTALL
)

# Remove post-action sensory action mutation. Doom already received the keypress by then.
env = re.sub(
    r'        # -----------------------------------------------------\n'
    r'        # Sensory emergency override\n'
    r'        # -----------------------------------------------------\n'
    r'        # Only override in serious cases\. Normal navigation should still\n'
    r'        # be handled by PPO \+ existing helpers for now\.\n'
    r'        # -----------------------------------------------------\n'
    r'        # Sensory emergency logging only\n'
    r'        # -----------------------------------------------------\n'
    r'        # SensoryModel can detect stuck/wall/spawn traps, but it should not\n'
    r'        # hijack PPO actions while we are trying to train a stable policy\.\n\n'
    r'        if self\.enable_sensory_action_override:\n'
    r'            sensory_recommended_action = sensory_state\.get\("recommended_action"\)\n'
    r'            action = sensory_recommended_action\n\n'
    r'            if action not in self\.get_allowed_actions\(\):\n'
    r'                action = "turn_right"\n\n'
    r'                print\(\n'
    r'                    f"\[override\] sensory_emergency: \{before\} -> \{action\} "\n'
    r'                    f"situation=\{sensory_state\[\'situation\'\]\}"\n'
    r'                \)\n\n'
    r'            game_state\["action"\] = action\n\n',
    '''        # -----------------------------------------------------
        # Sensory emergency logging only
        # -----------------------------------------------------
        # Do not mutate action here. Doom already received the keypress.
        # Real sensory emergency control happens pre-action.
        sensory_recommended_action = sensory_state.get("recommended_action")
        game_state["sensory_recommended_action"] = sensory_recommended_action

''',
    env,
    flags=re.DOTALL
)

env_path.write_text(env)

print("Patched scene_predictor.py, object_predictor.py, and doom_env.py")
