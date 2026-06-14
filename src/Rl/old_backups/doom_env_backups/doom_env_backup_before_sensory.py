import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    from debug.reward_debug_overlay import RewardDebugOverlay
except ImportError:
    RewardDebugOverlay = None

import frame_cache

from Helper.helper_bot import HelperBot
from controller.doom_controller import DoomController
from observation.observation_builder import ObservationBuilder
from rewards.reward_manager import RewardManager
from action.action_space import ActionSpace
from memory.exploration_memory import ExplorationMemory
from observation.frame_stack import FrameStack
from loggers.behavior_logger import BehaviorLogger
from loggers.trajectory_logger import TrajectoryLogger
from models.preference_model import PreferenceModel
from recording.clip_generator import ClipGenerator
from recording.smart_clip_generator import SmartClipGenerator
from recording.record_player_session import PlayerRecorder
from recording.event_recorder import EventRecorder
from curriculum.curriculum_manager import CurriculumManager
from observation.frame_processor import FrameProcessor
from navigation.checkpoint_tracker import CheckpointTracker
from navigation.level_guides import get_level_guide
from vision.scene_predictor import ScenePredictor

try:
    from observation.vision_detector import VisionDetector
except ImportError:
    VisionDetector = None

try:
    from combat.combat_tactics import CombatTactics
except ImportError:
    class CombatTactics:
        def choose_combat_action(self, state):
            return None


DOOM_BINARY = "/home/steven/Downloads/doomretro-master/build/doomretro"
DOOM_IWAD = "/usr/share/games/doom/freedoom1.wad"


class DoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, launch_doom=True, record=True):
        super().__init__()

        self.game_process = None

        if launch_doom:
            self.game_process = subprocess.Popen(
                [DOOM_BINARY, "-iwad", DOOM_IWAD],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Waiting 5 seconds for DOOM Retro to start...")
            time.sleep(5)
        else:
            print("Connecting to existing DOOM window...")

        # -----------------------------------------------------
        # Core systems
        # -----------------------------------------------------

        self.controller = DoomController()
        self.observer = ObservationBuilder()
        self.frame_stack = FrameStack(stack_size=3)
        self.frame_processor = FrameProcessor()
        self.exploration_memory = ExplorationMemory(map_size=512)
        self.reward_manager = RewardManager()
        self.actions = ActionSpace().ACTIONS
        self.preference_model = PreferenceModel()
        self.helper = HelperBot()
        self.curriculum = CurriculumManager()
        self.combat_tactics = CombatTactics()
        self.corner_trap_position = None
        self.corner_trap_steps = 0
        self.last_corner_escape_step = -100
        self.goal_turn_steps = 0
        self.max_goal_turn_steps = 6
        self.scene_predictor = None
        self.use_scene_classifier = True
        self.secret_use_locations = set()

        # -----------------------------------------------------
        # Sensory memory
        # -----------------------------------------------------
        # The scene classifier only sees one frame. These fields let the
        # environment remember whether the agent is repeatedly returning to
        # the same coordinate area, especially near the Freedoom spawn-room
        # boundary. This prevents fake "open_path" predictions from causing
        # endless forward movement into corners/walls.
        self.recent_position_tiles = []
        self.repeated_position_steps = 0
        self.last_sensory_scene_label = "unclear"
        self.last_sensory_scene_confidence = 0.0

        if VisionDetector is not None:
            self.vision_detector = VisionDetector(frame_processor=self.frame_processor)
        else:
            self.vision_detector = None
            print("[vision] observation.vision_detector not found — using built-in fallback vision.")

        self.checkpoint_tracker = CheckpointTracker(reach_radius=128.0)

        level_guide = get_level_guide("freedoom1_default")
        self.checkpoint_tracker.set_level_guide(
            checkpoints=level_guide["checkpoints"],
            secrets=level_guide["secrets"],
        )

        # -----------------------------------------------------
        # Corridor milestone tracking
        # -----------------------------------------------------
        # Freedoom places enemies early, so we do not require full level completion
        # before combat. First we count whether the agent can consistently reach
        # the first dangerous corridor/elevator area.
        self.corridor_reach_count = 0
        self.corridor_reached_this_episode = False
        self.corridor_reach_target = 3

        # -----------------------------------------------------
        # Penalty tuning
        # -----------------------------------------------------

        self.strong_penalty_mode = True
        self.penalty_decay_steps = 25_000

        # -----------------------------------------------------
        # Recording/debug systems
        # -----------------------------------------------------

        self.record = record
        self.reward_overlay = RewardDebugOverlay() if RewardDebugOverlay is not None else None
        self.event_recorder = EventRecorder()
        self.smart_clipper = SmartClipGenerator()

        # Turn off while debugging movement. Turn on later for dataset collection.
        self.collect_vision_frames = False
        self.vision_frame_interval = 1
        self.vision_frame_count = 0
        self.vision_dataset_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "vision_dataset",
            "raw_frames",
        )
        os.makedirs(self.vision_dataset_dir, exist_ok=True)

        if self.record:
            self.logger = BehaviorLogger()
            self.traj_logger = TrajectoryLogger()
            self.recorder = PlayerRecorder()
            self.clipper = ClipGenerator(clip_length=90)
        else:
            self.logger = None
            self.traj_logger = None
            self.recorder = None
            self.clipper = None

        if self.record and keyboard is not None and self.recorder is not None:
            self._kb_listener = keyboard.Listener(
                on_press=self.recorder.on_press,
                on_release=self.recorder.on_release,
            )
            self._kb_listener.start()
        else:
            self._kb_listener = None

        # -----------------------------------------------------
        # RL spaces
        # -----------------------------------------------------

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8,
        )

        # -----------------------------------------------------
        # Curriculum
        # -----------------------------------------------------

        self.curriculum_stage = 1
        self.max_stage = 3
        self.curriculum_rewards = []
        self.curriculum_log_interval = 50

        self.curriculum_thresholds = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.1,
            4: 0.1,
            5: 0.1,
            6: 0.2,
            7: 0.3,
        }

        # -----------------------------------------------------
        # Episode state
        # -----------------------------------------------------

        self._step_count = 0
        self._max_episode_steps = 1000
        self.episode_reward_total = 0.0
        self.best_episode_reward = -float("inf")

        self.stuck_counter = 0
        self.wall_contact_steps = 0
        self.last_wall_escape_step = -100
        self.route_progress_level = 0

        self.wall_escape_mode = False
        self.wall_escape_step = 0
        self.wall_escape_direction = "right"
        self.wall_escape_start_step = -100

        self.previous_health = None
        self._previous_frame = None
        self.last_player_position = None
        self.initial_distance = None
        self.closest_distance = None
        self.distance_traveled = 0.0
        self.visited_tiles = set()
        self.visited_areas = set()
        self.last_area_signature = None
        self.last_visual_signature = None
        self.visual_area_steps = 0
        self.prev_enemy_visible = False

        # -----------------------------------------------------
        # Aim/combat memory
        # -----------------------------------------------------

        self.last_target_signature = None
        self.same_target_shot_count = 0
        self.dead_target_ignore_steps = 0
        self.last_aim_assist_step = -100
        self.last_shot_step = -100

        self.consecutive_melee_steps = 0
        self.last_melee_step = -100
        self.consecutive_swap_steps = 0
        self.last_swap_step = -100

        # -----------------------------------------------------
        # Behavior counters
        # -----------------------------------------------------

        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.key_item_count = 0
        self.enemy_engagement_count = 0
        self.track_enemy_count = 0
        self.dodge_enemies_count = 0
        self.valid_shot_count = 0
        self.level_completion_count = 0
        self.enemy_visible_steps = 0
        self.swap_weapon_count = 0
        self.melee_attack_count = 0
        self.melee_close_bonus_count = 0
        self.weapon_pickup_count = 0
        self.ammo_pickup_count = 0
        self.enemy_kill_count = 0
        self.combat_survival_steps = 0
        self.dodge_when_damaged_count = 0
        self.retreat_from_close_enemy_count = 0
        self.episode_level_completions = 0

        self.mastery_level = 0
        self.final_mastery_level = 0
        self.final_mastery_rewards = []

        self._use_preference = False
        self._load_preference_model()

        if self.use_scene_classifier:
            try:
                self.scene_predictor = ScenePredictor()
            except Exception as e:
                print(f"[vision] Could not load scene classifier: {e}")
                self.scene_predictor = None

    # ---------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------

    def _load_preference_model(self):
        pref_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "checkpoints",
            "preference_model.pt",
        )

        if not os.path.exists(pref_path):
            print("No preference_model.pt found — skipping preference reward.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preference_model.load_state_dict(torch.load(pref_path, map_location=device))
        self.preference_model.eval()
        self._use_preference = True

    def get_stage_config(self):
        configs = {
            0: {
                "name": "movement_basic",
                "allow_shoot": False,
                "allow_melee": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
                "allow_use": False,
            },
            1: {
                "name": "movement_escape",
                "allow_shoot": False,
                "allow_melee": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
                "allow_use": False,
            },
            2: {
                "name": "doors_and_use",
                "allow_shoot": False,
                "allow_melee": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
                "allow_use": True,
            },
            3: {
                "name": "shoot_visible_enemies",
                "allow_shoot": True,
                "allow_melee": False,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "allow_use": True,
            },
            4: {
                "name": "combat_movement",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "allow_use": False,
            },
            5: {
                "name": "move_shoot_open_doors",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "allow_use": True,
            },
            6: {
                "name": "route_progress",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "allow_use": True,
            },
            7: {
                "name": "complete_level",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "allow_use": True,
            },
        }

        return configs.get(self.curriculum_stage, configs[7])

    def get_allowed_actions(self):
        stage = self.curriculum_stage

        if stage == 0:
            return [
                "move_forward",
                "turn_left",
                "turn_right",
            ]

        if stage == 1:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
            ]

        if stage == 2:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "use",
            ]

        if stage == 3:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "use",
                "shoot",
                "swap_weapon",
            ]

        if stage == 4:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "shoot",
                "use",
                "melee_attack",
                "swap_weapon",
            ]

        return [
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "shoot",
            "melee_attack",
            "use",
            "swap_weapon",
        ]

    def get_valid_actions(self):
        allowed = set(self.get_allowed_actions())
        return [i for i, action in enumerate(self.actions) if action in allowed]

    # ---------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.controller.release_all()
        self.prev_enemy_visible = False

        self._step_count = 0
        self.episode_reward_total = 0.0
        self.stuck_counter = 0
        self.wall_contact_steps = 0
        self.last_wall_escape_step = -100
        self.route_progress_level = 0
        self.goal_turn_steps = 0
        self.recent_position_tiles = []
        self.repeated_position_steps = 0
        self.last_sensory_scene_label = "unclear"
        self.last_sensory_scene_confidence = 0.0

        self.wall_escape_mode = False
        self.wall_escape_step = 0
        self.wall_escape_direction = "right"
        self.wall_escape_start_step = -100

        self.previous_health = None
        self._previous_frame = None
        self.last_player_position = None
        self.last_area_signature = None
        self.last_visual_signature = None
        self.visual_area_steps = 0
        self.distance_traveled = 0.0
        self.visited_tiles.clear()
        self.visited_areas.clear()
        self.corner_trap_position = None
        self.corner_trap_steps = 0
        self.last_corner_escape_step = -100

        self.last_target_signature = None
        self.same_target_shot_count = 0
        self.dead_target_ignore_steps = 0
        self.last_aim_assist_step = -100
        self.last_shot_step = -100

        self.consecutive_melee_steps = 0
        self.last_melee_step = -100
        self.consecutive_swap_steps = 0
        self.last_swap_step = -100

        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.key_item_count = 0
        self.enemy_engagement_count = 0
        self.track_enemy_count = 0
        self.dodge_enemies_count = 0
        self.valid_shot_count = 0
        self.level_completion_count = 0
        self.enemy_visible_steps = 0
        self.swap_weapon_count = 0
        self.melee_attack_count = 0
        self.melee_close_bonus_count = 0
        self.weapon_pickup_count = 0
        self.ammo_pickup_count = 0
        self.enemy_kill_count = 0
        self.combat_survival_steps = 0
        self.dodge_when_damaged_count = 0
        self.retreat_from_close_enemy_count = 0

        self.reward_manager.reset()
        self.checkpoint_tracker.reset()
        self.corridor_reached_this_episode = False

        # Do not clear self.secret_use_locations every episode.
        # This memory should persist across episodes.
        pass

        if hasattr(self.observer, "reset_tracking"):
            self.observer.reset_tracking()

        if hasattr(self.reward_manager, "enemy_detector"):
            if hasattr(self.reward_manager.enemy_detector, "reset"):
                self.reward_manager.enemy_detector.reset()

        frame_cache.reset_monitor()
        frame_cache.invalidate()

        wid = getattr(self.controller, "window_id", None)
        if wid:
            subprocess.call(
                ["xdotool", "windowfocus", "--sync", str(wid)],
                stderr=subprocess.DEVNULL,
            )

            time.sleep(0.2)

            for _ in range(3):
                subprocess.call(
                    ["xdotool", "key", "Return"],
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.25)

            self.controller.release_all()
            time.sleep(1.0)

        player_pos = self.observer.get_player_position()
        goal_pos = self.observer.get_goal_position()
        self.last_player_position = player_pos

        if player_pos is not None and goal_pos is not None:
            dist = float(np.linalg.norm(np.array(player_pos) - np.array(goal_pos)))
            self.initial_distance = dist
            self.closest_distance = dist
            self.reward_manager.initial_distance = dist
            self.reward_manager.closest_distance = dist
        else:
            self.initial_distance = None
            self.closest_distance = None

        config = self.get_stage_config()
        print(f"[Curriculum] Stage {self.curriculum_stage}: {config['name']}")

        frame = self.observer.build()
        observation = self.frame_stack.reset(frame)

        return observation, {}

    def step(self, action_index):
        frame_cache.invalidate()

        action_index = int(action_index)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        safe_frame = self.observer.build()
        safe_observation = self.frame_stack.add_frame(safe_frame)

        if action_index not in self.get_valid_actions():
            reward -= 0.2
            self.episode_reward_total += float(reward)
            return safe_observation, float(reward), terminated, truncated, info

        action = self.actions[action_index]
        config = self.get_stage_config()

        # -----------------------------------------------------
        # Pre-action perception
        # -----------------------------------------------------

        pre_frame = self.observer.get_frame()
        pre_game_state = self.observer.get_game_state()
        pre_vision = self.detect_vision(pre_frame)

        # -----------------------------------------------------
        # Pre-action learned scene prediction
        # -----------------------------------------------------
        # This must happen BEFORE perform_action().
        # Any action override after perform_action() only changes the log,
        # not the keypress sent to Doom.
        pre_scene_label = "unclear"
        pre_scene_confidence = 0.0
        pre_scene_probs = {}

        if self.scene_predictor is not None:
            pre_scene_result = self.scene_predictor.predict(pre_frame)
            pre_scene_label = pre_scene_result["label"]
            pre_scene_confidence = pre_scene_result["confidence"]
            pre_scene_probs = pre_scene_result["probs"]

            if self._step_count % 25 == 0:
                print(
                    f"[vision_model] label={pre_scene_label} "
                    f"conf={pre_scene_confidence:.2f}"
                )

        pre_enemy_visible = pre_vision["enemy_visible"]
        pre_enemy_centered = pre_vision["enemy_centered"]

        # In Stage 3+, the learned classifier is the authority for enemy
        # existence. The older color detector is only used for left/right/center
        # aiming hints after the classifier confirms an enemy.
        if self.curriculum_stage >= 3 and self.scene_predictor is not None:
            classifier_enemy = (
                pre_scene_label == "enemy"
                and pre_scene_confidence >= 0.85
            )

            if classifier_enemy:
                pre_enemy_visible = True
                pre_enemy_centered = pre_vision["enemy_centered"]
            else:
                pre_enemy_visible = False
                pre_enemy_centered = False
                pre_vision["enemy_left"] = False
                pre_vision["enemy_right"] = False
                pre_vision["enemy_confidence"] = 0.0

        # Stages 0-2 are navigation-only stages.
        # The current color-based enemy detector can be noisy, so we ignore enemy
        # labels until combat training starts at Stage 3.
        if self.curriculum_stage < 3:
            pre_enemy_visible = False
            pre_enemy_centered = False
            pre_vision["enemy_left"] = False
            pre_vision["enemy_right"] = False
            pre_vision["enemy_confidence"] = 0.0

        pre_game_state["curriculum_stage"] = self.curriculum_stage
        pre_game_state["enemy_visible"] = pre_enemy_visible
        pre_game_state["enemy_centered"] = pre_enemy_centered
        pre_game_state["enemy_left"] = pre_vision["enemy_left"]
        pre_game_state["enemy_right"] = pre_vision["enemy_right"]
        pre_game_state["enemy_close"] = pre_vision["enemy_confidence"] >= 120
        pre_game_state["door_visible"] = pre_vision["door_visible"]
        pre_game_state["door_centered"] = pre_vision["door_centered"]
        pre_game_state["front_wall_close"] = pre_vision["front_wall_close"]
        pre_game_state["left_wall_close"] = pre_vision["left_wall_close"]
        pre_game_state["right_wall_close"] = pre_vision["right_wall_close"]
        pre_game_state["stuck_counter"] = self.stuck_counter
        pre_game_state["distance_traveled"] = self.distance_traveled
        pre_game_state["wall_contact_steps"] = self.wall_contact_steps
        pre_game_state["scene_label"] = pre_scene_label
        pre_game_state["scene_confidence"] = pre_scene_confidence
        pre_game_state["scene_probs"] = pre_scene_probs
        pre_game_state["action"] = action

        # -----------------------------------------------------
        # Pre-action sensory override
        # -----------------------------------------------------
        # The neural classifier can call spawn-boundary/corner views open_path.
        # The sensory layer corrects that using position memory and wall ratios.
        pre_wall_info = self._wall_direction_info(pre_frame)
        pre_scene_label, pre_scene_confidence = self.sensory_scene_override(
            scene_label=pre_scene_label,
            scene_confidence=pre_scene_confidence,
            scene_probs=pre_scene_probs,
            game_state=pre_game_state,
            wall_info=pre_wall_info,
            distance_moved=None,
            motion=None,
            action=action,
        )

        pre_game_state["scene_label"] = pre_scene_label
        pre_game_state["scene_confidence"] = pre_scene_confidence

        # -----------------------------------------------------
        # Action correction stack
        # -----------------------------------------------------

        original_action = action

        # 1) Sanitize the PPO action for the current stage.
        action = self.sanitize_action(action, pre_game_state, pre_enemy_visible)
        if action != original_action:
            print(f"[override] sanitize: {original_action} -> {action}")

        pre_game_state["action"] = action

        # 2) Stage 0/1/2 are movement/door learning only.
        # Combat helpers must not hijack these stages.
        if self.curriculum_stage < 3:
            before = action
            action = self.exploration_assist_action(
                action=action,
                enemy_visible=False,
                distance_moved=None,
                motion=None,
            )
            if action != before:
                print(f"[override] exploration: {before} -> {action}")

            before = action
            action = self.wall_assist_action(
                action=action,
                frame=pre_frame,
                enemy_visible=False,
                distance_moved=None,
                motion=None,
            )
            if action != before:
                print(f"[override] wall: {before} -> {action}")

            if self.curriculum_stage == 2:
                before = action
                action = self.door_assist_action(
                    action=action,
                    frame=pre_frame,
                    enemy_visible=False,
                    distance_moved=None,
                    motion=None,
                )
                if action != before:
                    print(f"[override] door: {before} -> {action}")

        # 3) Combat stages: aim first, then combat, then movement/route helpers.
        else:
            aimed_this_step = False

            # Aim alignment gets priority over shooting/swap logic.
            if pre_enemy_visible and not pre_enemy_centered:
                if pre_game_state.get("enemy_left", False):
                    before = action
                    action = "turn_left"
                    aimed_this_step = True
                    print(f"[override] aim_priority: {before} -> {action}")

                elif pre_game_state.get("enemy_right", False):
                    before = action
                    action = "turn_right"
                    aimed_this_step = True
                    print(f"[override] aim_priority: {before} -> {action}")

            if not aimed_this_step:
                before = action
                pre_game_state["action"] = action
                tactical_action = None #self.combat_tactics.choose_combat_action(pre_game_state)

                if tactical_action is not None:
                    # If route stages have made no progress, do not replace
                    # all movement with shoot/melee/swap.
                    if self.curriculum_stage >= 5 and self.distance_traveled < 25.0:
                        if tactical_action in ["shoot", "melee_attack", "swap_weapon"]:
                            tactical_action = "move_forward"

                    # Anti-melee-spam rule.
                    if tactical_action == "melee_attack" and self.consecutive_melee_steps >= 3:
                        if self.stuck_counter >= 3:
                            tactical_action = "move_backward"
                        else:
                            tactical_action = (
                                "strafe_left"
                                if (self._step_count // 5) % 2 == 0
                                else "strafe_right"
                            )

                    action = tactical_action
                    if action != before:
                        print(f"[override] combat_tactics: {before} -> {action}")

                else:
                    before = action
                    action = self.aim_assist_action(
                        action=action,
                        frame=pre_frame,
                        enemy_visible=pre_enemy_visible,
                        enemy_centered=pre_enemy_centered,
                        ammo=pre_game_state.get("ammo", 0),
                    )
                    if action != before:
                        print(f"[override] aim_assist: {before} -> {action}")

            before = action
            action = self.exploration_assist_action(
                action=action,
                enemy_visible=pre_enemy_visible,
                distance_moved=None,
                motion=None,
            )
            if action != before:
                print(f"[override] exploration: {before} -> {action}")

            before = action
            action = self.wall_assist_action(
                action=action,
                frame=pre_frame,
                enemy_visible=pre_enemy_visible,
                distance_moved=None,
                motion=None,
            )
            if action != before:
                print(f"[override] wall: {before} -> {action}")

            before = action
            action = self.door_assist_action(
                action=action,
                frame=pre_frame,
                enemy_visible=pre_enemy_visible,
                distance_moved=None,
                motion=None,
            )
            if action != before:
                print(f"[override] door: {before} -> {action}")


        # -----------------------------------------------------
        # Pre-action learned-vision safety and combat
        # -----------------------------------------------------
        # These happen BEFORE perform_action(), so they affect the real keypress.

        # pre_wall_info was computed during the sensory override above.

        before_goal = action
        action = self.goal_assist_action(
            action=action,
            game_state=pre_game_state,
            wall_info=pre_wall_info,
            enemy_visible=pre_enemy_visible,
        )
        if action != before_goal:
            print(f"[override] goal_assist: {before_goal} -> {action}")

        if (
            pre_scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall"]
            and pre_scene_confidence >= 0.55
            and action == "move_forward"
        ):
            before = action

            if pre_scene_label == "obstacle":
                # Low walls / half walls often require sidestepping,
                # not just backing up forever.
                if self._step_count % 4 in [0, 1]:
                    action = "move_backward"
                elif self._step_count % 4 == 2:
                    action = "strafe_left"
                else:
                    action = "strafe_right"
            elif pre_scene_label == "boundary_or_stuck_wall":
                # Spawn boundaries and corner loops need a stronger escape cycle.
                if self._step_count % 6 in [0, 1]:
                    action = "move_backward"
                elif self._step_count % 6 in [2, 3]:
                    action = "turn_right"
                else:
                    action = "strafe_right"
            else:
                action = "move_backward"

            print(
                f"[override] vision_blocker: {before} -> {action} "
                f"label={pre_scene_label} conf={pre_scene_confidence:.2f}"
            )

        # Corner/stuck escape beats combat. Do not shoot while trapped.
        if self.corner_trap_steps >= 20:
            before = action
            cycle = self.corner_trap_steps % 12

            if cycle in [0, 1, 2]:
                action = "move_backward"
            elif cycle in [3, 4, 5]:
                action = "turn_right"
            elif cycle in [6, 7]:
                action = "strafe_right"
            elif cycle in [8, 9]:
                action = "turn_left"
            else:
                action = "move_forward"

            print(
                f"[override] corner_escape: {before} -> {action} "
                f"corner_steps={self.corner_trap_steps}"
            )
        else:
            before_vision_combat = action
            action = self.vision_combat_action(
                action=action,
                scene_label=pre_scene_label,
                scene_confidence=pre_scene_confidence,
                health=pre_game_state.get("health", 100),
                ammo=pre_game_state.get("ammo", 0),
            )

            if action != before_vision_combat:
                print(
                    f"[override] vision_combat: "
                    f"{before_vision_combat} -> {action} "
                    f"label={pre_scene_label} conf={pre_scene_confidence:.2f}"
                )

        # -----------------------------------------------------
        # Pre-action wall bubble safety
        # -----------------------------------------------------
        # This runs BEFORE perform_action(), so it affects the real keypress.
        #
        # The post-action wall bubble is still useful for rewards/debugging,
        # but action overrides must happen before Doom receives input.
        pre_wall_bubble = self.wall_bubble_state(
            wall_info=pre_wall_info,
            distance_moved=None,
            motion=None,
            action=action,
        )

        before_wall_bubble = action
        action = self.wall_bubble_action(action, pre_wall_bubble)

        if action != before_wall_bubble:
            print(
                f"[override] pre_wall_bubble: {before_wall_bubble} -> {action} "
                f"level={pre_wall_bubble['level']} "
                f"dir={pre_wall_bubble['direction']} "
                f"L={pre_wall_bubble['left']:.2f} "
                f"F={pre_wall_bubble['front']:.2f} "
                f"R={pre_wall_bubble['right']:.2f}"
            )

        # Track melee streak after final helper choice.
        if action == "melee_attack":
            self.consecutive_melee_steps += 1
            self.last_melee_step = self._step_count
        else:
            self.consecutive_melee_steps = 0

        # Prevent endless swap_weapon loops BEFORE executing the action.
        if action == "swap_weapon":
            self.consecutive_swap_steps += 1
            self.last_swap_step = self._step_count

            if self.consecutive_swap_steps > 3:
                before = action

                if pre_enemy_visible and not pre_enemy_centered:
                    if pre_game_state.get("enemy_left", False):
                        action = "turn_left"
                    elif pre_game_state.get("enemy_right", False):
                        action = "turn_right"
                    else:
                        action = "move_backward"
                elif self.stuck_counter >= 3:
                    action = "move_backward"
                else:
                    action = "move_forward"

                print(f"[override] swap_spam: {before} -> {action}")
        else:
            self.consecutive_swap_steps = 0

        if self.stuck_counter >= 8:
            cycle = self._step_count % 8

            before = action

            if cycle in [0, 1]:
                action = "move_backward"
            elif cycle in [2, 3]:
                action = "turn_right"
            elif cycle == 4:
                action = "strafe_right"
            elif cycle == 5:
                action = "turn_left"
            else:
                action = "move_forward"

            print(f"[override] hard_stuck_escape: {before} -> {action}")

        # Final safety pass BEFORE action execution.
        final_before = action
        action = self.sanitize_action(action, pre_game_state, pre_enemy_visible)

        if action != final_before:
            print(f"[override] final_sanitize: {final_before} -> {action}")

        if action not in self.get_allowed_actions():
            # Safer fallback than always moving forward.
            if pre_wall_info.get("front_wall", False) or pre_wall_info.get("front_ratio", 0.0) > 0.45:
                fixed_action = "move_backward"
            else:
                fixed_action = "move_forward"

            print(
                f"[override] final_safety: {action} -> {fixed_action} "
                f"because stage={self.curriculum_stage} allowed={self.get_allowed_actions()}"
            )
            action = fixed_action

        helper_action = self.helper.get_action(pre_game_state)

        if action == helper_action:
            reward += 0.05
        else:
            reward -= 0.01

        # -----------------------------------------------------
        # Execute action
        # -----------------------------------------------------

        self.perform_action(action)

        # -----------------------------------------------------
        # Post-action perception
        # -----------------------------------------------------

        raw_frame = self.observer.get_frame()
        vision = self.detect_vision(raw_frame)

        _, motion, _, _ = self.frame_processor.extract(raw_frame)
        self._save_vision_frame(raw_frame)

        enemy_visible = vision["enemy_visible"]
        enemy_centered = vision["enemy_centered"]

        # Ignore noisy enemy vision before combat stages.
        if self.curriculum_stage < 3:
            enemy_visible = False
            enemy_centered = False
            vision["enemy_confidence"] = 0.0

        if enemy_visible:
            self.enemy_visible_steps += 1

        floor_green_ratio = self.frame_processor.floor_green_ratio(raw_frame)
        red_ratio = self.frame_processor.red_flash_ratio(raw_frame)

        game_state = self.observer.get_game_state()

        game_state["enemy_visible"] = enemy_visible
        game_state["enemy_centered"] = enemy_centered
        game_state["motion"] = motion
        game_state["action"] = action
        game_state["curriculum_stage"] = self.curriculum_stage

        distance_moved = self._update_position_tracking(
            game_state,
            frame=raw_frame,
            motion=motion,
            action=action,
        )

        game_state["distance_moved"] = distance_moved
        corner_trapped = self.detect_corner_trap(game_state)
        game_state["corner_trapped"] = corner_trapped

        # -----------------------------------------------------
        # Post-action scene state
        # -----------------------------------------------------
        # Use the pre-action prediction for reward/state consistency.
        # Do not modify action here; Doom already received the keypress.
        scene_label = pre_scene_label
        scene_confidence = pre_scene_confidence
        scene_probs = pre_scene_probs

        scene_label, scene_confidence = self.sensory_scene_override(
            scene_label=scene_label,
            scene_confidence=scene_confidence,
            scene_probs=scene_probs,
            game_state=game_state,
            wall_info=None,
            distance_moved=distance_moved,
            motion=motion,
            action=action,
        )

        game_state["scene_label"] = scene_label
        game_state["scene_confidence"] = scene_confidence
        game_state["scene_probs"] = scene_probs

        # -----------------------------------------------------
        # Corridor milestone
        # -----------------------------------------------------
        # The first corridor is the first real danger checkpoint.
        # Reaching it consistently means the movement/route policy is good enough
        # to begin learning basic combat.
        corridor_reached = self.detect_corridor_reached(game_state)
        game_state["corridor_reached"] = corridor_reached

        if corridor_reached and not self.corridor_reached_this_episode:
            self.corridor_reached_this_episode = True
            self.corridor_reach_count += 1

            reward += 5.0
            self.reward_manager.add("corridor_reached", 5.0)

            print(
                f"[milestone] corridor reached "
                f"{self.corridor_reach_count}/{self.corridor_reach_target}"
            )

        # Corner escape action changes already happen before perform_action().
        # Do not mutate action here after Doom has already received input.

        # -----------------------------------------------------
        # Current state values
        # -----------------------------------------------------

        current_weapon = str(game_state.get("weapon", "")).lower()
        ammo = game_state.get("ammo", 0)
        ammo_delta = game_state.get("ammo_delta", 0)
        weapon_delta = game_state.get("weapon_delta", 0)
        kill_delta = game_state.get("kill_delta", 0)
        health = game_state.get("health", 100)
        health_delta = game_state.get("health_delta", 0)

        if self.curriculum_stage >= 3 and health > 0:
            self.combat_survival_steps += 1

        enemy_close = (
            enemy_visible
            and enemy_centered
            and motion > 0.5
        )

        # -----------------------------------------------------
        # Wall/contact detection
        # -----------------------------------------------------

        wall_contact = (
            action == "move_forward"
            and motion < 1.0
            and distance_moved <= 0.1
        )

        if wall_contact:
            self.wall_contact_steps += 1
        else:
            self.wall_contact_steps = max(0, self.wall_contact_steps - 1)

        wall_info = self._wall_direction_info(raw_frame)

        game_state["left_wall_ratio"] = wall_info["left_ratio"]
        game_state["front_wall_ratio"] = wall_info["front_ratio"]
        game_state["right_wall_ratio"] = wall_info["right_ratio"]
        # Reward open-space movement and penalize wall-hugging.
        # This is the part that teaches the agent not to use walls as rails.
        wall_space_reward = self.wall_proximity_penalty(wall_info, action)
        reward += wall_space_reward
        if wall_space_reward < 0:
            self.reward_manager.add("wall_hugging_penalty", wall_space_reward)
        elif wall_space_reward > 0:
            self.reward_manager.add("wall_avoidance_reward", wall_space_reward)

        open_reward = self.open_space_reward(wall_info, distance_moved, action)
        reward += open_reward
        if open_reward > 0:
            self.reward_manager.add("open_space_movement", open_reward)

        enemy_confidence = vision.get("enemy_confidence", 0.0)
        spacing_reward = self.enemy_spacing_reward(
            enemy_visible=enemy_visible,
            enemy_centered=enemy_centered,
            enemy_confidence=enemy_confidence,
            action=action,
            health_delta=health_delta,
        )
        reward += spacing_reward
        if spacing_reward < 0:
            self.reward_manager.add("bad_enemy_spacing", spacing_reward)
        elif spacing_reward > 0:
            self.reward_manager.add("good_enemy_spacing", spacing_reward)

        checkpoint_reward = 0.0
        checkpoint_info = {
            "distance_to_checkpoint": None,
            "checkpoint_name": None,
            "checkpoint_reached": False,
            "distance_to_secret": None,
            "nearest_secret_name": None,
            "secret_reached": False,
            "all_checkpoints_reached": False,
        }

        game_state["distance_to_checkpoint"] = checkpoint_info.get("distance_to_checkpoint")
        game_state["checkpoint_name"] = checkpoint_info.get("checkpoint_name")
        game_state["checkpoint_reached"] = checkpoint_info.get("checkpoint_reached")
        game_state["distance_to_secret"] = checkpoint_info.get("distance_to_secret")
        game_state["nearest_secret_name"] = checkpoint_info.get("nearest_secret_name")
        game_state["secret_reached"] = checkpoint_info.get("secret_reached")
        game_state["all_checkpoints_reached"] = checkpoint_info.get("all_checkpoints_reached")

        should_fight = enemy_visible and ammo > 0
        should_dodge = enemy_visible and health_delta < 0
        should_retreat = enemy_visible and enemy_centered and health < 45
        should_escape_wall = self.wall_contact_steps >= 2 or self.stuck_counter > 12

        game_state["wall_contact"] = wall_contact
        game_state["should_fight"] = should_fight
        game_state["should_dodge"] = should_dodge
        game_state["should_retreat"] = should_retreat
        game_state["should_escape_wall"] = should_escape_wall

        wall_info = self._wall_direction_info(raw_frame)

        game_state["left_wall_ratio"] = wall_info["left_ratio"]
        game_state["front_wall_ratio"] = wall_info["front_ratio"]
        game_state["right_wall_ratio"] = wall_info["right_ratio"]

        # -----------------------------------------------------
        # Wall proximity bubble
        # -----------------------------------------------------
        # This gives the agent a simple safety field:
        # green = safe, orange = too close, red = bumping/trapped.
        wall_bubble = self.wall_bubble_state(
            wall_info=wall_info,
            distance_moved=distance_moved,
            motion=motion,
            action=action,
        )

        game_state["wall_bubble_level"] = wall_bubble["level"]
        game_state["wall_bubble_direction"] = wall_bubble["direction"]

        # Post-action bubble is for reward/debug only. Action overrides must happen
        # before perform_action(), otherwise the log changes but the keypress does not.
        suggested_bubble_action = self.wall_bubble_action(action, wall_bubble)

        if suggested_bubble_action != action:
            print(
                f"[post_wall_bubble] would prefer {action} -> {suggested_bubble_action} "
                f"level={wall_bubble['level']} dir={wall_bubble['direction']} "
                f"L={wall_bubble['left']:.2f} F={wall_bubble['front']:.2f} R={wall_bubble['right']:.2f}"
            )

        if wall_bubble["level"] == "red":
            reward += self.add_penalty("red_wall_bubble", -1.5)

        elif wall_bubble["level"] == "orange":
            reward += self.add_penalty("orange_wall_bubble", -0.4)

        elif wall_bubble["level"] == "green":
            if distance_moved > 2.0 and action in ["move_forward", "strafe_left", "strafe_right"]:
                reward += 0.15
                self.reward_manager.add("green_space_movement", 0.15)

        if scene_label in ["obstacle", "boundary_or_stuck_wall"] and scene_confidence >= 0.55:
            if action == "move_forward":
                reward += self.add_penalty("push_into_obstacle", -1.5)

            elif action in ["move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right"]:
                reward += 0.35
                self.reward_manager.add("avoid_obstacle", 0.35)
        # -----------------------------------------------------
        # Tactical behavior rewards
        # -----------------------------------------------------

        if wall_contact:
            reward += self.add_penalty("wall_contact", -1.0)

        if wall_info["front_wall"] and action == "move_forward":
            reward += self.add_penalty("push_into_front_wall", -1.0)

        if wall_contact and action == "move_forward":
            reward += self.add_penalty("push_wall_forward", -1.5)

        if self.stuck_counter >= 8:
            reward += self.add_penalty("stuck_near_wall", -1.0)

        if self.stuck_counter > 15:
            reward += self.add_penalty("stuck_serious", -1.0)

        if self.stuck_counter > 25:
            reward += self.add_penalty("stuck_critical", -2.0)

        if wall_info["front_wall"] and action == "move_backward":
            reward += 0.4
            self.reward_manager.add("back_away_from_wall", 0.4)

        if wall_info["left_wall"] and action in ["turn_right", "strafe_right"]:
            reward += 0.25
            self.reward_manager.add("escape_left_wall", 0.25)

        if wall_info["right_wall"] and action in ["turn_left", "strafe_left"]:
            reward += 0.25
            self.reward_manager.add("escape_right_wall", 0.25)

        if enemy_visible and action == "shoot" and ammo > 0:
            reward += 0.25
            self.reward_manager.add("shoot_enemy_on_sight", 0.25)

        if enemy_visible and health_delta < 0 and action in ["strafe_left", "strafe_right"]:
            reward += 0.8
            self.dodge_when_damaged_count += 1
            self.reward_manager.add("dodge_when_damaged", 0.8)

        if enemy_visible and health < 45 and action == "move_backward":
            reward += 0.6
            self.retreat_from_close_enemy_count += 1
            self.reward_manager.add("retreat_low_health", 0.6)

        if not enemy_visible and action == "shoot":
            reward += self.add_penalty("shoot_without_enemy", -2.0)

        if not enemy_visible and action in ["turn_left", "turn_right"] and self.stuck_counter < 5:
            reward += self.add_penalty("unneeded_turning", -0.6)

        if self.wall_escape_mode and distance_moved > 4.0 and motion > 2.0:
            reward += 0.5
            self.reward_manager.add("successful_wall_escape", 0.5)

        # -----------------------------------------------------
        # Stage-specific movement/combat reward
        # -----------------------------------------------------

        if self.curriculum_stage >= 5:
            if distance_moved > 3.0:
                reward += 0.15
                self.reward_manager.add("route_movement_progress", 0.15)

            if action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]:
                reward += 0.03
                self.reward_manager.add("route_movement_action", 0.03)

            if action == "melee_attack" and self.consecutive_melee_steps > 3:
                reward += self.add_penalty("melee_spam", -1.0)

            if action == "swap_weapon" and self.consecutive_swap_steps > 3:
                reward += self.add_penalty("swap_spam", -1.0)

            if self.distance_traveled < 25.0 and self._step_count > 100:
                reward += self.add_penalty("no_route_progress", -1.5)

        if self.corner_trap_steps >= 60:
            reward -= 5.0
            terminated = True
            info["corner_trap_reset"] = True
            print("[reset] corner trap timeout")

        # -----------------------------------------------------
        # Debug prints
        # -----------------------------------------------------

        if self._step_count % 50 == 0:
            print(
                f"[tactical] action={action} "
                f"enemy={enemy_visible} centered={enemy_centered} "
                f"ammo={ammo} health={health} "
                f"motion={motion:.2f} dist={distance_moved:.2f} "
                f"wall={self.wall_contact_steps} stuck={self.stuck_counter} "
                f"hits={self.valid_shot_count} kills={self.enemy_kill_count} "
                f"doors={self.door_interaction_count} pickups={self.pickup_count}"
            )

            print(
                f"[wall] left={wall_info['left_ratio']:.2f} "
                f"front={wall_info['front_ratio']:.2f} "
                f"right={wall_info['right_ratio']:.2f}"
            )

        if self._step_count % 25 == 0:
            print(
                f"[coords] "
                f"x={game_state.get('x'):.1f} "
                f"y={game_state.get('y'):.1f}"
            )

        if self._step_count % 100 == 0:
            print(
                f"[pos] x={game_state.get('x')} "
                f"y={game_state.get('y')} "
                f"shared={game_state.get('shared_state_available')}"
            )

            print(
                f"[nav] checkpoint={checkpoint_info.get('checkpoint_name')} "
                f"dist={checkpoint_info.get('distance_to_checkpoint')} "
                f"secret={checkpoint_info.get('nearest_secret_name')} "
                f"secret_dist={checkpoint_info.get('distance_to_secret')} "
                f"reward={checkpoint_reward:.2f}"
            )

        # -----------------------------------------------------
        # Resource / combat event rewards
        # -----------------------------------------------------

        if ammo_delta > 0:
            reward += 0.4
            self.ammo_pickup_count += 1
            self.reward_manager.add("ammo_pickup", 0.4)

        if weapon_delta > 0:
            reward += 3.0
            self.weapon_pickup_count += 1
            self.reward_manager.add("weapon_pickup", 3.0)

        if kill_delta > 0:
            reward += 5.0 * kill_delta
            self.enemy_kill_count += kill_delta
            self.reward_manager.add("enemy_kill", 5.0 * kill_delta)

        if enemy_visible and health_delta < 0 and action in ["strafe_left", "strafe_right"]:
            reward += 0.5
            self.dodge_when_damaged_count += 1

        if enemy_visible and enemy_close and action == "move_backward":
            reward += 0.4
            self.retreat_from_close_enemy_count += 1

        # -----------------------------------------------------
        # Core movement / stuck rewards
        # -----------------------------------------------------

        if motion < 1.5:
            self.stuck_counter += 1
        else:
            if self.stuck_counter > 10:
                reward += 1.0
            self.stuck_counter = 0

        if motion < 1.0 and action == "move_forward":
            reward += self.add_penalty("low_motion_forward", -0.3)

        if self.stuck_counter > 10 and action in [
            "turn_left",
            "turn_right",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]:
            reward += 0.2
            self.reward_manager.add("escape_stuck_action", 0.2)

        if self.stuck_counter > 25:
            reward += self.add_penalty("long_stuck", -1.0)

        if distance_moved > 5.0:
            reward += 0.04
            self.movement_count += 1

        # -----------------------------------------------------
        # Resource / exploration rewards
        # -----------------------------------------------------

        self.reward_manager.update_resource_reward(
            game_state.get("health", 0),
            game_state.get("ammo", 0),
        )

        if game_state.get("health_delta", 0) > 0 or game_state.get("ammo_delta", 0) > 0:
            reward += 0.5
            self.pickup_count += 1
            self.key_item_count += 1

        # -----------------------------------------------------
        # Stage 3 combat progression
        # -----------------------------------------------------

        if self.curriculum_stage == 3:
            movement_actions = [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
                "shoot",
                "use",
                "swap_weapon",
            ]

            if not enemy_visible and action == "move_forward":
                reward += 0.03

            if distance_moved > 5.0:
                reward += 0.04

            if len(self.visited_tiles) > 0 and action in movement_actions:
                reward += 0.01

            if action in ["turn_left", "turn_right"] and not enemy_visible:
                reward += self.add_penalty("stage3_unneeded_turn", -0.1)

            if action in ["turn_left", "turn_right"] and enemy_visible:
                reward += 0.05

            if enemy_visible and not self.prev_enemy_visible:
                reward += 2.0
                self.reward_manager.add("enemy_discovered", 2.0)

            if enemy_visible:
                reward += 0.50

            if enemy_visible and enemy_centered:
                reward += 1.00
                self.track_enemy_count += 1

            if action == "shoot" and enemy_visible and ammo > 0:
                # Smaller reward to avoid ammo-spam learning.
                reward += 0.30

                if enemy_centered:
                    reward += 0.50
                    self.enemy_engagement_count += 1
                    self.valid_shot_count += 1

            if action == "shoot" and not enemy_visible:
                reward += self.add_penalty("stage3_shoot_without_enemy", -2.0)

        elif self.curriculum_stage >= 4:
            if action == "shoot":
                if not config.get("allow_shoot", False):
                    reward += self.add_penalty("shoot_not_allowed", -2.5)

                elif ammo <= 0:
                    reward += self.add_penalty("shoot_no_ammo", -2.0)

                elif config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
                    reward += self.add_penalty("shoot_without_visible_enemy", -2.0)

                else:
                    if enemy_visible:
                        reward += 0.6
                        self.enemy_engagement_count += 1

                    if enemy_visible:
                        reward += 1.2
                        self.valid_shot_count += 1
                        self.reward_manager.add("visible_enemy_shot", 1.2)

                        if enemy_centered:
                            reward += 0.8
                            self.reward_manager.add("centered_enemy_shot", 0.8)

                    if enemy_visible and not enemy_centered:
                        reward += 0.15

                    if ammo <= 5:
                        reward += self.add_penalty("low_ammo_shot_pressure", -0.5)

        # -----------------------------------------------------
        # Melee/use/swap rewards
        # -----------------------------------------------------

        if action == "melee_attack":
            self.melee_attack_count += 1

            if not config.get("allow_melee", False):
                reward += self.add_penalty("melee_not_allowed", -1.0)

            elif not enemy_visible:
                reward += self.add_penalty("melee_without_enemy", -0.4)

            elif enemy_visible and enemy_centered:
                reward += 0.8
                self.reward_manager.add("melee_enemy_centered", 0.8)

                if enemy_close:
                    reward += 1.5
                    self.melee_close_bonus_count += 1
                    self.reward_manager.add("melee_close_range", 1.5)

            else:
                reward += self.add_penalty("bad_melee_spacing", -0.2)

        if action == "use":
            # -------------------------------------------------
            # Door / button / secret use logic
            # -------------------------------------------------
            # IMPORTANT:
            # meaningful_use must be calculated BEFORE we check it.
            #
            # This fixes:
            # UnboundLocalError: cannot access local variable
            # 'meaningful_use' where it is not associated with a value.
            #
            # The agent only gets credit for pressing use when:
            #   - use is allowed in the current stage
            #   - a door/button is likely visible and centered
            #   - the agent is close enough / slowed enough for use to matter
            door_visible = vision.get("door_visible", False)
            door_centered = vision.get("door_centered", False)

            # The learned scene classifier can also help identify doors/buttons.
            if scene_label == "door_or_button" and scene_confidence >= 0.45:
                door_visible = True

                # If the classifier sees a door/button, treat it as centered enough
                # for now. Later you can improve this with object localization.
                door_centered = True

            meaningful_use = (
                self.curriculum_stage in [2, 3, 5, 6, 7]
                and door_visible
                and door_centered
                and (
                    self.stuck_counter >= 2
                    or self.wall_contact_steps >= 1
                    or distance_moved < 2.0
                    or scene_label == "door_or_button"
                )
            )

            if meaningful_use:
                self.door_interaction_count += 1
                reward += 1.0
                self.reward_manager.add("meaningful_door_use", 1.0)

                x = game_state.get("x")
                y = game_state.get("y")

                if x is not None and y is not None:
                    use_tile = (int(float(x) // 64), int(float(y) // 64))

                    # Remember useful use locations across episodes.
                    # This helps with secret doors/buttons because the agent
                    # can discover that use worked at this approximate tile.
                    if use_tile not in self.secret_use_locations:
                        self.secret_use_locations.add(use_tile)

                        reward += 3.0
                        self.reward_manager.add("new_use_location_discovered", 3.0)

                        print(f"[memory] useful use location discovered: {use_tile}")

            else:
                reward += self.add_penalty("use_spam_penalty", -1.0)


        if action == "swap_weapon":
            self.swap_weapon_count += 1
            if enemy_visible and ammo > 5:
                reward += self.add_penalty("unneeded_weapon_swap_combat", -0.3)
            elif not enemy_visible and ammo > 5:
                reward += self.add_penalty("unneeded_weapon_swap", -0.2)

        # -----------------------------------------------------
        # Weapon quality awareness
        # -----------------------------------------------------

        if self.curriculum_stage >= 4:
            using_weak_weapon = (
                "pistol" in current_weapon
                or current_weapon == ""
            )

            if using_weak_weapon and self._step_count > 300:
                reward += self.add_penalty("weak_weapon_pressure", -0.02)

            if self.weapon_pickup_count > 0:
                reward += 0.05

        # -----------------------------------------------------
        # Route progress milestones
        # -----------------------------------------------------

        if self.curriculum_stage >= 7:
            if game_state.get("checkpoint_reached", False):
                reward += 2.0
                self.reward_manager.add("stage7_checkpoint_bonus", 2.0)

            if game_state.get("secret_reached", False):
                reward += 5.0
                self.reward_manager.add("stage7_secret_bonus", 5.0)

            if game_state.get("all_checkpoints_reached", False):
                reward += 20.0
                self.reward_manager.add("all_checkpoints_reached", 20.0)

            if len(self.visited_tiles) >= 10 and self.route_progress_level < 1:
                reward += 5.0
                self.route_progress_level = 1
                self.reward_manager.add("route_tiles_10", 5.0)

            if self.pickup_count >= 2 and self.route_progress_level < 2:
                reward += 5.0
                self.route_progress_level = 2
                self.reward_manager.add("route_pickups_2", 5.0)

            if self.door_interaction_count >= 1 and self.route_progress_level < 3:
                reward += 8.0
                self.route_progress_level = 3
                self.reward_manager.add("route_door_1", 8.0)

            if self.enemy_kill_count >= 1 and self.route_progress_level < 4:
                reward += 5.0
                self.route_progress_level = 4
                self.reward_manager.add("route_kill_1", 5.0)

            if len(self.visited_tiles) >= 25 and self.route_progress_level < 5:
                reward += 8.0
                self.route_progress_level = 5
                self.reward_manager.add("route_tiles_25", 8.0)

        # -----------------------------------------------------
        # Damage/environment detection
        # -----------------------------------------------------

        if hasattr(self.reward_manager, "detect_acid_damage"):
            self.reward_manager.detect_acid_damage(
                game_state.get("health_delta", 0),
                distance_moved,
                floor_green_ratio,
            )

        if hasattr(self.reward_manager, "detect_barrel_damage"):
            self.reward_manager.detect_barrel_damage(
                game_state.get("health_delta", 0),
                red_ratio,
            )

        self.reward_manager.update_stagnation_penalty(self.stuck_counter)
        reward += self.reward_manager.get_reward()

        self.prev_enemy_visible = enemy_visible


        # -----------------------------------------------------
        # Termination
        # -----------------------------------------------------

        health = game_state.get("health", 100)

        # Emergency combat override:
        # Freedoom has enemies very early. If the agent is still in Stage 2 but
        # taking damage, allow simple defensive shooting instead of dying helplessly.
        if self.curriculum_stage == 2 and health_delta < 0 and ammo > 0:
            if enemy_visible and enemy_centered:
                reward += 0.5
                self.reward_manager.add("emergency_enemy_defense", 0.5)

        death_like_screen = (
            health <= 0
            or (
                health < 5
                and motion < 0.5
                and self.stuck_counter > 15
            )
        )

        if game_state.get("level_complete", False):
            reward += 100.0
            self.level_completion_count += 1
            terminated = True

        if death_like_screen:
            reward -= 25.0
            self.reward_manager.add("death_penalty", -25.0)
            terminated = True
            info["death"] = True

        if self.stuck_counter >= 35:
            reward -= 2.0
            terminated = True
            info["stuck_reset"] = True

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps

        if terminated or truncated:
            self.controller.release_all()

        processed_frame = self.observer.build()
        observation = self.frame_stack.add_frame(processed_frame)

        if self.record:
            self._record_step(raw_frame, action, reward, game_state)

        self.curriculum_rewards.append(reward)

        if len(self.curriculum_rewards) > 100:
            self.curriculum_rewards.pop(0)

        self.update_curriculum()

        self.episode_reward_total += float(reward)

        if terminated or truncated:
            if self.curriculum_stage >= 7:
                if self.episode_reward_total > self.best_episode_reward:
                    self.best_episode_reward = self.episode_reward_total
                    print(
                        f"[Mastery] New best episode reward: "
                        f"{self.best_episode_reward:.2f}"
                    )

        return observation, float(reward), terminated, truncated, info

    # ---------------------------------------------------------
    # Action execution
    # ---------------------------------------------------------

    def perform_action(self, action):
        allowed = self.get_allowed_actions()

        if self._step_count % 10 == 0:
            print(
                f"[perform_action] step={self._step_count} "
                f"stage={self.curriculum_stage} "
                f"action={action} "
                f"allowed={action in allowed} "
                f"window_id={getattr(self.controller, 'window_id', None)}"
            )

        if action not in allowed:
            print(f"[perform_action] BLOCKED action={action} allowed={allowed}")
            return

        self.controller.release_all()

        if action == "move_forward":
            self.controller.hold_key("w", duration=0.12)

        elif action == "move_backward":
            self.controller.hold_key("s", duration=0.12)

        elif action == "turn_left":
            self.controller.hold_key("Left", duration=0.06)

        elif action == "turn_right":
            self.controller.hold_key("Right", duration=0.06)

        elif action == "strafe_left":
            self.controller.hold_key("a", duration=0.12)

        elif action == "strafe_right":
            self.controller.hold_key("d", duration=0.12)

        elif action == "shoot":
            self.controller.shoot()

        elif action == "melee_attack":
            self.controller.shoot()

        elif action == "use":
            self.controller.use()

        elif action == "swap_weapon":
            self.controller.swap_weapon()

    # ---------------------------------------------------------
    # Vision helpers
    # ---------------------------------------------------------

    def detect_vision(self, frame):
        if self.vision_detector is not None:
            result = self.vision_detector.detect(frame)

            # Guarantee expected keys exist even if the external detector is incomplete.
            fallback = self._fallback_vision(frame)
            for key, value in fallback.items():
                result.setdefault(key, value)

            return result

        return self._fallback_vision(frame)

    def _fallback_vision(self, frame):
        enemy_error, enemy_confidence = self._enemy_horizontal_error(frame)
        door_error, door_confidence = self._door_horizontal_error(frame)
        wall_info = self._wall_direction_info(frame)

        return {
            "enemy_visible": enemy_confidence >= 20,
            "enemy_centered": abs(enemy_error) < 0.18 and enemy_confidence >= 20,
            "enemy_left": enemy_error < -0.18 and enemy_confidence >= 20,
            "enemy_right": enemy_error > 0.18 and enemy_confidence >= 20,
            "enemy_confidence": float(enemy_confidence),
            "enemy_x": float(enemy_error) if enemy_confidence >= 20 else None,
            "enemy_y": None,

            "door_visible": door_confidence >= 250,
            "door_centered": abs(door_error) < 0.15 and door_confidence >= 250,
            "door_left": door_error < -0.20 and door_confidence >= 250,
            "door_right": door_error > 0.20 and door_confidence >= 250,
            "door_confidence": float(door_confidence),

            "pickup_visible": False,
            "weapon_visible": False,
            "ammo_visible": False,
            "health_visible": False,

            "projectile_visible": False,
            "barrel_visible": False,
            "exit_visible": False,

            "front_wall_close": wall_info["front_wall"],
            "left_wall_close": wall_info["left_wall"],
            "right_wall_close": wall_info["right_wall"],
            "front_wall_ratio": wall_info["front_ratio"],
            "left_wall_ratio": wall_info["left_ratio"],
            "right_wall_ratio": wall_info["right_ratio"],

            "scene_label": None,
            "scene_confidence": 0.0,

            "depth_center": None,
            "depth_left": None,
            "depth_right": None,
        }

    # ---------------------------------------------------------
    # Tracking / recording helpers
    # ---------------------------------------------------------

    def _visual_area_signature(self, frame):
        if frame is None or frame.size == 0:
            return None

        small = frame[::8, ::8, :].astype(np.int16)
        quantized = small // 32

        return hash(quantized.tobytes())

    def _update_position_tracking(self, game_state, frame=None, motion=0.0, action=None):
        distance_moved = 0.0

        movement_action = action in [
            "move_forward",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]

        if not game_state.get("shared_state_available", False):
            if action in ["turn_left", "turn_right"] and motion > 2.0:
                self.reward_manager.add("spin_without_translation", -0.05)

            if movement_action and motion > 2.0:
                distance_moved = float(motion)
                self.distance_traveled += distance_moved

            signature = self._visual_area_signature(frame)

            if movement_action and signature is not None and signature != self.last_visual_signature:
                self.last_visual_signature = signature
                self.visual_area_steps += 1

                if self.visual_area_steps % 5 == 0:
                    pseudo_tile = ("visual", self.visual_area_steps // 5)

                    if pseudo_tile not in self.visited_tiles:
                        self.visited_tiles.add(pseudo_tile)
                        self.exploration_count += 1
                        self.reward_manager.add("visual_exploration", 0.05)

            return distance_moved

        player_x = game_state.get("x")
        player_y = game_state.get("y")

        if player_x is None or player_y is None:
            return distance_moved

        current_player_pos = (player_x, player_y)

        if self.last_player_position is not None:
            distance_moved = float(
                np.linalg.norm(
                    np.array(current_player_pos) - np.array(self.last_player_position)
                )
            )

        self.last_player_position = current_player_pos
        self.distance_traveled += distance_moved

        tile = (int(player_x // 64), int(player_y // 64))

        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            self.exploration_count += 1

        self.exploration_memory.visit(player_x, player_y)
        novelty_reward = self.exploration_memory.get_novelty(player_x, player_y) * 0.3
        self.reward_manager.add("exploration_novelty", novelty_reward)

        goal_pos = self.observer.get_goal_position()

        if goal_pos is not None:
            current_dist = float(
                np.linalg.norm(np.array(current_player_pos) - np.array(goal_pos))
            )

            if self.closest_distance is None or current_dist < self.closest_distance:
                self.closest_distance = current_dist
                self.reward_manager.closest_distance = current_dist

        return distance_moved

    def preference_reward(self, frame):
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            score = self.preference_model(frame_tensor)

        return float(score.item())

    def _record_step(self, frame, action, reward, game_state):
        if self.reward_overlay is not None:
            frame = self.reward_overlay.draw(frame, self.reward_manager.breakdown)

        if self.logger is not None:
            self.logger.log(
                frame,
                action,
                game_state.get("health", 0),
                game_state.get("ammo", 0),
            )

        if self.traj_logger is not None:
            self.traj_logger.record(frame, action, reward)

        if self.recorder is not None:
            self.recorder.record_frame(frame)

        if self.event_recorder is not None:
            self.event_recorder.record_frame(frame)

        if self.clipper is not None:
            self.clipper.record(frame, action)

        if self.smart_clipper is not None:
            self.smart_clipper.observe(frame, action)

    def _save_vision_frame(self, frame, game_state=None, action=None):
        if not self.collect_vision_frames:
            return

        if frame is None or frame.size == 0:
            return

        if self._step_count % self.vision_frame_interval != 0:
            return

        path = os.path.join(
            self.vision_dataset_dir,
            f"frame_{self.vision_frame_count:06d}.jpg",
        )

        cv2.imwrite(path, frame)
        self.vision_frame_count += 1

        if self.vision_frame_count % 50 == 0:
            print(f"[vision] saved {self.vision_frame_count} frames")

    # ---------------------------------------------------------
    # Curriculum
    # ---------------------------------------------------------

    def update_curriculum(self):
        if len(self.curriculum_rewards) < 50:
            return

        avg_reward = sum(self.curriculum_rewards) / len(self.curriculum_rewards)
        threshold = self.curriculum_thresholds.get(self.curriculum_stage, 0.0)

        reward_ready = True if threshold <= 0.0 else avg_reward > threshold
        behavior_ready = False
        should_log = self._step_count % self.curriculum_log_interval == 0

        if self.curriculum_stage == 0:
            behavior_ready = (
                self.distance_traveled >= 100.0
                and self.movement_count >= 10
            )

            if should_log:
                print(
                    f"  [Stage 0] movement_basic | "
                    f"distance: {self.distance_traveled:.1f}/100 | "
                    f"moves: {self.movement_count}/10 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 1:
            behavior_ready = (
                self.distance_traveled >= 200.0
                and len(self.visited_tiles) >= 3
                and self.stuck_counter < 10
            )

            if should_log:
                print(
                    f"  [Stage 1] movement_escape | "
                    f"distance: {self.distance_traveled:.1f}/200 | "
                    f"tiles: {len(self.visited_tiles)}/3 | "
                    f"stuck: {self.stuck_counter}/10 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 2:
            behavior_ready = (
                self.corridor_reach_count >= self.corridor_reach_target
            )

            if should_log:
                print(
                    f"  [Stage 2] reach_corridor | "
                    f"corridor: {self.corridor_reach_count}/{self.corridor_reach_target} | "
                    f"distance: {self.distance_traveled:.1f} | "
                    f"doors/use: {self.door_interaction_count} | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 3:
            # Do not advance just because the agent spammed shots.
            # Require either an actual kill, or survival with limited,
            # controlled shooting and some ammo remaining.
            behavior_ready = (
                self.enemy_kill_count >= 1
                or (
                    self.combat_survival_steps >= 250
                    and self.valid_shot_count >= 5
                    and self.valid_shot_count <= 80
                )
            )

            if should_log:
                print(
                    f"  [Stage 3] corridor_combat_survival | "
                    f"valid_shots: {self.valid_shot_count}/3 | "
                    f"kills: {self.enemy_kill_count}/1 | "
                    f"survival: {self.combat_survival_steps}/100 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 4:
            combat_hits = self.valid_shot_count + self.melee_close_bonus_count

            behavior_ready = (
                (
                    combat_hits >= 4
                    and self.enemy_visible_steps >= 8
                )
                or self.enemy_kill_count >= 1
                or self.dodge_when_damaged_count >= 2
                or self.retreat_from_close_enemy_count >= 2
            )

            if should_log:
                print(
                    f"  [Stage 4] combat_movement | "
                    f"hits/melee: {combat_hits}/4 | "
                    f"visible: {self.enemy_visible_steps}/8 | "
                    f"kills: {self.enemy_kill_count}/1 | "
                    f"dodges: {self.dodge_when_damaged_count}/2 | "
                    f"retreats: {self.retreat_from_close_enemy_count}/2 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 5:
            behavior_ready = (
                self.distance_traveled >= 250.0
                and len(self.visited_tiles) >= 4
                and (
                    self.door_interaction_count >= 1
                    or self.enemy_kill_count >= 1
                    or self.valid_shot_count >= 3
                )
            )

            if should_log:
                print(
                    f"  [Stage 5] move_shoot_open_doors | "
                    f"distance: {self.distance_traveled:.1f}/250 | "
                    f"doors: {self.door_interaction_count}/1 | "
                    f"hits: {self.valid_shot_count}/3 | "
                    f"kills: {self.enemy_kill_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 6:
            behavior_ready = (
                self.distance_traveled >= 600.0
                and len(self.visited_tiles) >= 8
                and (
                    self.pickup_count >= 1
                    or self.door_interaction_count >= 1
                    or self.enemy_kill_count >= 1
                )
            )

            if should_log:
                print(
                    f"  [Stage 6] route_progress | "
                    f"distance: {self.distance_traveled:.1f}/600 | "
                    f"tiles: {len(self.visited_tiles)}/8 | "
                    f"pickups: {self.pickup_count}/1 | "
                    f"doors: {self.door_interaction_count}/1 | "
                    f"kills: {self.enemy_kill_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage >= 7:
            if should_log:
                print(
                    f"  [Stage 7] complete_level | "
                    f"tiles: {len(self.visited_tiles)} | "
                    f"distance: {self.distance_traveled:.1f} | "
                    f"doors: {self.door_interaction_count} | "
                    f"pickups: {self.pickup_count} | "
                    f"hits: {self.valid_shot_count + self.melee_close_bonus_count} | "
                    f"kills: {self.enemy_kill_count} | "
                    f"complete: {self.level_completion_count} | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

            return

        if reward_ready and behavior_ready and self.curriculum_stage < self.max_stage:
            old_stage = self.curriculum_stage
            self.curriculum_stage += 1
            self.curriculum_rewards.clear()
            self._reset_stage_counters()

            print(
                f"[Curriculum] ADVANCING FROM STAGE "
                f"{old_stage} → {self.curriculum_stage}"
            )

    # ---------------------------------------------------------
    # Combat / action helpers
    # ---------------------------------------------------------

    def _enemy_horizontal_error(self, frame):
        if frame is None or frame.size == 0:
            return 0.0, 0

        if not hasattr(self.frame_processor, "enemy_heatmap"):
            return 0.0, 0

        heatmap = self.frame_processor.enemy_heatmap(frame)

        if heatmap is None or heatmap.size == 0:
            return 0.0, 0

        h, w = heatmap.shape
        gameplay_heatmap = heatmap[: int(h * 0.78), :]

        _, xs = np.where(gameplay_heatmap > 0.5)

        confidence = len(xs)

        if confidence < 8:
            return 0.0, confidence

        enemy_x = float(np.mean(xs))
        center_x = w / 2.0
        aim_error = float((enemy_x - center_x) / center_x)

        return aim_error, confidence

    def _target_signature(self, frame):
        aim_error, confidence = self._enemy_horizontal_error(frame)

        if confidence < 20:
            return None

        return round(aim_error, 1)


    def aim_assist_action(self, action, frame, enemy_visible, enemy_centered, ammo):
        """
        Aim helper only.

        This function is intentionally NOT allowed to choose shoot anymore.
        The old version returned shoot from too many branches, which caused:
        - random shooting
        - ammo drain
        - valid_shot_count going up without kills
        - shoot/melee/move_forward override loops

        Shooting is handled only by vision_combat_action().
        """
        if self.curriculum_stage < 3:
            return action

        if self.dead_target_ignore_steps > 0:
            self.dead_target_ignore_steps -= 1
            return action

        if not enemy_visible:
            return action

        ammo = int(ammo or 0)

        # No ammo means do not aim-shoot. Back up / strafe instead.
        if ammo <= 0:
            if self._step_count % 2 == 0:
                return "move_backward"
            return "strafe_right"

        aim_error, confidence = self._enemy_horizontal_error(frame)

        # If the old color detector is weak/noisy, do not shoot.
        # The learned scene classifier decides enemy existence.
        if confidence < 12:
            return "move_backward"

        if aim_error < -0.18:
            self.last_aim_assist_step = self._step_count
            return "turn_left"

        if aim_error > 0.18:
            self.last_aim_assist_step = self._step_count
            return "turn_right"

        # Already roughly aimed; leave shoot decision to vision_combat_action().
        return action

    def exploration_assist_action(self, action, enemy_visible, distance_moved=None, motion=None):
        if enemy_visible:
            return action

        if action in ["shoot", "melee_attack"]:
            return "move_forward"

        if self.wall_contact_steps >= 2 or self.stuck_counter > 12:
            cycle = self._step_count % 6

            if cycle == 0:
                return "move_backward"
            if cycle == 1:
                return "turn_right"
            if cycle == 2:
                return "turn_right"
            if cycle == 3:
                return "strafe_left"
            if cycle == 4:
                return "move_forward"

            return "turn_left"

        if action in ["turn_left", "turn_right"]:
            if self._step_count - self.last_aim_assist_step > 10:
                return "move_forward"

        if (
            distance_moved is not None
            and motion is not None
            and distance_moved <= 0.0
            and motion < 1.0
        ):
            return "move_forward"

        return action

    def penalty_scale(self):
        if not self.strong_penalty_mode:
            return 1.0

        progress = min(1.0, self._step_count / float(self.penalty_decay_steps))
        return 4.0 - (3.0 * progress)

    def add_penalty(self, name, base_penalty):
        scaled_penalty = base_penalty * self.penalty_scale()
        self.reward_manager.add(name, scaled_penalty)
        return scaled_penalty

    def door_assist_action(self, action, frame, enemy_visible, distance_moved=None, motion=None):
        if self.stuck_counter >= 6 or self.wall_contact_steps >= 2 or self.corner_trap_steps >= 12:
            return action
        
        if enemy_visible:
            return action

        if self.curriculum_stage not in [2, 5, 6, 7]:
            return action

        door_error, door_confidence = self._door_horizontal_error(frame)

        if door_confidence < 250:
            return action

        if motion is not None and motion > 2.0:
            return action

        if abs(door_error) < 0.15:
            if self.stuck_counter >= 4 or self.wall_contact_steps >= 2:
                return "use"

            return action

        if self.stuck_counter >= 4:
            if door_error < -0.20:
                return "turn_left"

            if door_error > 0.20:
                return "turn_right"

        return action

    def _door_horizontal_error(self, frame):
        if frame is None or frame.size == 0:
            return 0.0, 0

        if len(frame.shape) == 3 and frame.shape[0] in [3, 9] and frame.shape[-1] not in [3, 4]:
            frame = np.transpose(frame[:3], (1, 2, 0))

        h, w = frame.shape[:2]
        gameplay = frame[: int(h * 0.70), :]

        blue = gameplay[:, :, 0].astype(np.int16)
        green = gameplay[:, :, 1].astype(np.int16)
        red = gameplay[:, :, 2].astype(np.int16)

        brown = (
            (red > 55)
            & (green > 35)
            & (blue < 120)
            & (red >= green)
        )

        gray = (
            (red > 50)
            & (green > 50)
            & (blue > 50)
            & (np.abs(red - green) < 25)
            & (np.abs(green - blue) < 25)
        )

        mask = brown | gray

        center_left = int(w * 0.20)
        center_right = int(w * 0.80)

        mask[:, :center_left] = False
        mask[:, center_right:] = False

        _, xs = np.where(mask)

        confidence = len(xs)

        if confidence < 20:
            return 0.0, confidence

        door_x = float(np.mean(xs))
        center_x = w / 2.0
        error = float((door_x - center_x) / center_x)

        return error, confidence
    
    def sensory_scene_override(
        self,
        scene_label,
        scene_confidence,
        scene_probs=None,
        game_state=None,
        wall_info=None,
        distance_moved=None,
        motion=None,
        action=None,
    ):
        """
        Combine the learned scene classifier with body feedback.

        Why this exists:
        - The classifier sees one image and may call spawn/corner views open_path.
        - The environment knows whether the agent actually moved.
        - If x/y repeats, motion is low, or wall ratios are high, we should treat
          that "open_path" as a boundary/stuck situation.

        This is the first sensory model: vision + position + movement feedback.
        """
        label = scene_label or "unclear"
        confidence = float(scene_confidence or 0.0)
        game_state = game_state or {}
        scene_probs = scene_probs or {}

        x = game_state.get("x")
        y = game_state.get("y")
        tile = None

        if x is not None and y is not None:
            try:
                tile = (int(float(x) // 64), int(float(y) // 64))
            except Exception:
                tile = None

        # Update repeated-position memory only after an action when movement data exists.
        if distance_moved is not None or motion is not None:
            if tile is not None:
                self.recent_position_tiles.append(tile)
                if len(self.recent_position_tiles) > 12:
                    self.recent_position_tiles.pop(0)

                repeats = self.recent_position_tiles.count(tile)
                if repeats >= 5:
                    self.repeated_position_steps += 1
                else:
                    self.repeated_position_steps = max(0, self.repeated_position_steps - 1)

        front_ratio = 0.0
        side_ratio = 0.0
        if wall_info is not None:
            front_ratio = float(wall_info.get("front_ratio", 0.0))
            side_ratio = max(
                float(wall_info.get("left_ratio", 0.0)),
                float(wall_info.get("right_ratio", 0.0)),
            )

        stuck_by_body = (
            self.stuck_counter >= 5
            or self.wall_contact_steps >= 2
            or self.corner_trap_steps >= 12
            or self.repeated_position_steps >= 3
        )

        no_translation = (
            action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]
            and distance_moved is not None
            and motion is not None
            and distance_moved <= 0.5
            and motion < 1.5
        )

        # Coordinates from the user's logs repeatedly show spawn/boundary trouble
        # around x=-496 and y=80. Keep this as a soft rule, not a hard map hack.
        near_spawn_boundary = False
        if x is not None and y is not None:
            try:
                near_spawn_boundary = float(x) <= -490.0 or float(y) <= 82.0
            except Exception:
                near_spawn_boundary = False

        # Correct fake-open predictions.
        if label == "open_path":
            if stuck_by_body or no_translation or near_spawn_boundary:
                label = "boundary_or_stuck_wall"
                confidence = max(confidence, 0.90)
            elif front_ratio >= 0.55:
                label = "front_wall"
                confidence = max(confidence, 0.75)
            elif side_ratio >= 0.85:
                label = "obstacle"
                confidence = max(confidence, 0.70)

        # Very low-confidence enemy in navigation stages should not hijack movement.
        # Your current dataset is enemy-heavy, so be conservative outside combat.
        if self.curriculum_stage < 3 and label == "enemy" and confidence < 0.90:
            if scene_probs.get("obstacle", 0.0) >= 0.25:
                label = "obstacle"
                confidence = max(float(scene_probs.get("obstacle", 0.0)), 0.60)
            else:
                label = "unclear"
                confidence = max(confidence, 0.40)

        self.last_sensory_scene_label = label
        self.last_sensory_scene_confidence = confidence
        return label, confidence

    def detect_corner_trap(self, game_state):
        """
        Detect when the agent keeps returning to the same small coordinate area.

        This is different from basic wall contact. The agent may have motion,
        but still be trapped in a corner loop.
        """
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return False

        current_tile = (int(x // 32), int(y // 32))

        if self.corner_trap_position is None:
            self.corner_trap_position = current_tile
            self.corner_trap_steps = 0
            return False

        if current_tile == self.corner_trap_position:
            self.corner_trap_steps += 1
        else:
            self.corner_trap_position = current_tile
            self.corner_trap_steps = 0

        return self.corner_trap_steps >= 20

    def wall_assist_action(self, action, frame, enemy_visible, distance_moved=None, motion=None):
        if action == "move_backward" and self.stuck_counter < 8 and self.corner_trap_steps < 12:
            return action
        
        if enemy_visible and self.stuck_counter < 6 and self.wall_contact_steps < 2:
            return action

        if self.curriculum_stage < 1:
            return action

        wall_info = self._wall_direction_info(frame)

        # Wall assist should only choose safer actions.
        # Reward/penalty math happens inside step(), where reward exists.
        front_wall = wall_info["front_wall"]
        left_ratio = wall_info["left_ratio"]
        right_ratio = wall_info["right_ratio"]
        front_ratio = wall_info["front_ratio"]

        blocked = (
            self.wall_contact_steps >= 2
            or self.stuck_counter >= 5
        )

        if (
            action == "move_forward"
            and motion is not None
            and distance_moved is not None
            and motion < 1.0
            and distance_moved <= 0.2
        ):
            blocked = True

        if action == "move_forward" and front_wall and front_ratio > 0.60 and self.stuck_counter >= 2:
            blocked = True

        if blocked and not self.wall_escape_mode:
            self.wall_escape_mode = True
            self.wall_escape_step = 0
            self.wall_escape_start_step = self._step_count

            if left_ratio > right_ratio + 0.05:
                self.wall_escape_direction = "right"
            elif right_ratio > left_ratio + 0.05:
                self.wall_escape_direction = "left"
            else:
                self.wall_escape_direction = (
                    "right" if (self._step_count // 20) % 2 == 0 else "left"
                )

        if self.wall_escape_mode:
            if (
                distance_moved is not None
                and distance_moved > 4.0
                and motion is not None
                and motion > 2.0
                and not front_wall
            ):
                self.wall_escape_mode = False
                self.wall_escape_step = 0
                return action

            if self._step_count - self.wall_escape_start_step > 18:
                self.wall_escape_mode = False
                self.wall_escape_step = 0
                return "move_forward"

            cycle = self.wall_escape_step % 6
            self.wall_escape_step += 1

            rotate_action = "turn_right" if self.wall_escape_direction == "right" else "turn_left"
            strafe_action = "strafe_right" if self.wall_escape_direction == "right" else "strafe_left"

            if cycle == 0:
                return "move_backward"

            if cycle == 1:
                return rotate_action

            if cycle == 2:
                return rotate_action

            if cycle == 3:
                return strafe_action

            if cycle == 4:
                return rotate_action

            return "move_forward"

        return action

    def _wall_direction_info(self, frame):
        default = {
            "left_ratio": 0.0,
            "front_ratio": 0.0,
            "right_ratio": 0.0,
            "left_wall": False,
            "front_wall": False,
            "right_wall": False,
        }

        if frame is None or frame.size == 0:
            return default

        if len(frame.shape) == 3 and frame.shape[0] in [3, 9] and frame.shape[-1] not in [3, 4]:
            frame = np.transpose(frame[:3], (1, 2, 0))

        h, w = frame.shape[:2]
        gameplay = frame[: int(h * 0.70), :]

        blue = gameplay[:, :, 0].astype(np.int16)
        green = gameplay[:, :, 1].astype(np.int16)
        red = gameplay[:, :, 2].astype(np.int16)

        brown = (
            (red > 45)
            & (green > 25)
            & (blue < 130)
            & (red >= green)
            & (green >= blue * 0.45)
        )

        gray = (
            (red > 45)
            & (green > 45)
            & (blue > 45)
            & (np.abs(red - green) < 30)
            & (np.abs(green - blue) < 30)
        )

        wall_mask = brown | gray

        left_region = wall_mask[:, : int(w * 0.33)]
        front_region = wall_mask[:, int(w * 0.33): int(w * 0.66)]
        right_region = wall_mask[:, int(w * 0.66):]

        left_ratio = float(np.mean(left_region))
        front_ratio = float(np.mean(front_region))
        right_ratio = float(np.mean(right_region))

        return {
            "left_ratio": left_ratio,
            "front_ratio": front_ratio,
            "right_ratio": right_ratio,
            "left_wall": left_ratio > 0.42,
            "front_wall": front_ratio > 0.42,
            "right_wall": right_ratio > 0.42,
        }
    
    def wall_bubble_state(self, wall_info, distance_moved=None, motion=None, action=None):
        """
        Convert wall ratios into a green/orange/red proximity bubble.

        Important:
        - Pre-action calls do not know distance_moved or motion yet.
        - So pre-action bubble must NOT treat distance_moved=0 as bumping.
        - Bumping should only be detected after the action has actually happened.
        """
        left = float(wall_info.get("left_ratio", 0.0))
        front = float(wall_info.get("front_ratio", 0.0))
        right = float(wall_info.get("right_ratio", 0.0))

        side = max(left, right)

        # Only detect bumping when we have real post-action movement data.
        has_motion_data = distance_moved is not None and motion is not None

        bumping_forward = (
            has_motion_data
            and action == "move_forward"
            and distance_moved <= 0.5
            and motion < 1.5
        )

        # Red should mean serious danger, not normal side wall proximity.
        front_blocked = front >= 0.65
        side_cramped = side >= 0.85

        if front_blocked or bumping_forward:
            level = "red"
        elif side_cramped or front >= 0.45 or side >= 0.60:
            level = "orange"
        else:
            level = "green"

        if front >= max(left, right):
            direction = "front"
        elif left > right:
            direction = "left"
        else:
            direction = "right"

        return {
            "level": level,
            "direction": direction,
            "left": left,
            "front": front,
            "right": right,
            "side": side,
            "front_blocked": front_blocked,
            "side_cramped": side_cramped,
            "bumping_forward": bumping_forward,
        }
    
    def wall_bubble_action(self, action, bubble):
        """
        Safer wall-bubble correction.

        Orange does not mean panic. It only nudges the agent away.
        Red means blocked/bumping and should force a stronger escape.
        """
        level = bubble["level"]
        direction = bubble["direction"]

        if level == "green":
            return action

        # Red = blocked or actually bumping.
        if level == "red":
            if direction == "front":
                if action == "move_forward":
                    return "move_backward"

                # Rotate away after backing up.
                if bubble["left"] > bubble["right"]:
                    return "turn_right"
                return "turn_left"

            if direction == "left":
                if action in ["move_forward", "strafe_left"]:
                    return "strafe_right"
                return "turn_right"

            if direction == "right":
                if action in ["move_forward", "strafe_right"]:
                    return "strafe_left"
                return "turn_left"

        # Orange = close to wall, but not necessarily stuck.
        # Do not override turning/use/shoot unless it is clearly bad.
        if level == "orange":
            if direction == "front" and action == "move_forward":
                return "move_backward"

            if direction == "left" and action == "strafe_left":
                return "strafe_right"

            if direction == "right" and action == "strafe_right":
                return "strafe_left"

        return action
    
    def enemy_spacing_reward(self, enemy_visible, enemy_centered, enemy_confidence, action, health_delta):
        """
        Encourage keeping distance from close enemies.
        High enemy_confidence usually means the enemy occupies a large part of the screen,
        so it is probably close.
        """
        reward = 0.0

        enemy_close = enemy_visible and enemy_confidence >= 120
        enemy_very_close = enemy_visible and enemy_confidence >= 220

        if not enemy_visible:
            return 0.0

        # Walking into a visible/centered enemy is dangerous.
        if enemy_centered and action == "move_forward":
            reward -= 0.75

        # Very close enemies should trigger retreat or strafe.
        if enemy_close and action in ["move_backward", "strafe_left", "strafe_right"]:
            reward += 0.35

        if enemy_very_close and action == "move_forward":
            reward -= 1.25

        # If taking damage, reward evasive movement.
        if health_delta < 0 and action in ["move_backward", "strafe_left", "strafe_right"]:
            reward += 0.75

        # Standing close and using/shooting without repositioning can be risky.
        if enemy_very_close and action == "use":
            reward -= 0.50

        return reward

    def wall_proximity_penalty(self, wall_info, action):
        """
        Penalize hugging walls even if the agent is technically moving.
        We want the agent to use open space, not scrape along walls.
        """
        penalty = 0.0

        left = wall_info["left_ratio"]
        front = wall_info["front_ratio"]
        right = wall_info["right_ratio"]

        side_wall = max(left, right)

        # Strong side-wall hugging penalty.
        if side_wall > 0.55:
            penalty -= 0.20

        if side_wall > 0.75:
            penalty -= 0.45

        # Front wall danger.
        if front > 0.45 and action == "move_forward":
            penalty -= 0.60

        # Strafing into a wall is bad.
        if left > 0.55 and action == "strafe_left":
            penalty -= 0.50

        if right > 0.55 and action == "strafe_right":
            penalty -= 0.50

        # Turning away from a wall is good.
        if left > 0.55 and action == "turn_right":
            penalty += 0.20

        if right > 0.55 and action == "turn_left":
            penalty += 0.20

        return penalty

    def open_space_reward(self, wall_info, distance_moved, action):
        """
        Reward movement through open space instead of scraping along walls.

        The agent should learn that distance only counts as useful movement
        when there is enough open space around it.
        """
        left = wall_info["left_ratio"]
        front = wall_info["front_ratio"]
        right = wall_info["right_ratio"]
        side_wall = max(left, right)

        if distance_moved > 3.0 and front < 0.25 and side_wall < 0.35:
            if action in ["move_forward", "strafe_left", "strafe_right"]:
                return 0.20

        return 0.0
    
    def _angle_to_goal(self, game_state):
        """
        Calculate the angle from the player's current position to the current goal.

        This uses the shared-memory x/y position and the observer's goal position.
        It returns None if we do not have enough information.
        """
        x = game_state.get("x")
        y = game_state.get("y")
        angle = game_state.get("angle")

        goal = self.observer.get_goal_position()

        if x is None or y is None or angle is None or goal is None:
            return None

        goal_x, goal_y = goal

        dx = goal_x - x
        dy = goal_y - y

        # Angle from player to goal in degrees.
        target_angle = np.degrees(np.arctan2(dy, dx))

        # Normalize both angles to 0-360.
        current_angle = float(angle) % 360.0
        target_angle = float(target_angle) % 360.0

        # Shortest signed angular difference: -180 to +180.
        diff = (target_angle - current_angle + 180.0) % 360.0 - 180.0

        return diff

    def goal_assist_action(self, action, game_state, wall_info=None, enemy_visible=False):
        """
        Navigation helper.

        If the agent is wandering, this nudges it toward the current goal:
        - large angle error -> rotate toward goal
        - small angle error -> move forward
        - nearby wall -> let wall logic handle it instead

        This keeps the agent from drifting aimlessly around the map.
        """

        # Wall, obstacle, and stuck escape must beat goal steering.
        if self.stuck_counter >= 3 or self.wall_contact_steps >= 1 or self.corner_trap_steps >= 10:
            return action

        if game_state.get("scene_label") in ["front_wall", "obstacle", "boundary_or_stuck_wall"]:
            if game_state.get("scene_confidence", 0.0) >= 0.50:
                return action

        if self.corner_trap_steps >= 10 or self.stuck_counter >= 5:
            return action

        # Do not goal-steer while the agent is trapped.
        # Corner/wall escape must win, or goal assist will keep rotating the agent
        # back into the same corner.
        if self.corner_trap_steps >= 12 or self.wall_escape_mode or self.stuck_counter >= 8:
            return action
        
        if enemy_visible and self.curriculum_stage >= 3:
            return action

        # Goal steering is useful from Stage 2 onward.
        # Stage 1 should still focus mostly on raw movement/escape.
        if self.curriculum_stage < 2:
            return action

        if wall_info is not None:
            front_wall = wall_info.get("front_wall", False)
            front_ratio = wall_info.get("front_ratio", 0.0)
            side_wall = max(
                wall_info.get("left_ratio", 0.0),
                wall_info.get("right_ratio", 0.0),
            )

            # If we are too close to walls, do not force goal movement.
            # Wall escape should take priority.
            if front_wall or front_ratio > 0.45 or side_wall > 0.70:
                return action

        angle_error = self._angle_to_goal(game_state)

        if angle_error is None:
            return action

        # Do not let goal assist hijack every frame.
        if self._step_count % 3 != 0:
            return action

        # If we have been rotating too long, force forward movement briefly.
        if self.goal_turn_steps >= self.max_goal_turn_steps:
            self.goal_turn_steps = 0
            return "move_forward"

        # If facing far away from the goal, rotate first.
        if angle_error < -25.0:
            self.goal_turn_steps += 1
            return "turn_left"

        if angle_error > 25.0:
            self.goal_turn_steps += 1
            return "turn_right"

        # If roughly facing the goal, move forward and reset turn counter.
        if abs(angle_error) <= 25.0:
            self.goal_turn_steps = 0
            return "move_forward"

        return action
    
    def detect_corridor_reached(self, game_state):
        """
        Detect whether the agent reached the early dangerous corridor/elevator area.

        These coordinates are based on the logs you showed. The agent repeatedly
        reaches around x=800-950 and y=450-600 before enemy damage becomes serious.

        You can tune these numbers later after collecting more coordinates.
        """
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return False

        return (
            750.0 <= float(x) <= 980.0
            and 420.0 <= float(y) <= 650.0
        )
    

    def vision_combat_action(self, action, scene_label, scene_confidence, health, ammo):
        """
        The ONLY helper that may turn an action into shoot.

        Rules:
        - Never shoot before Stage 3.
        - Never shoot with no ammo.
        - Never shoot while stuck/corner trapped.
        - Only shoot when the learned scene classifier is confident.
        - Fire short bursts, then move to keep distance.
        """
        if self.curriculum_stage < 3:
            return action

        # Corner/stuck escape beats combat.
        if self.corner_trap_steps >= 15 or self.stuck_counter >= 5:
            return action

        enemy_seen = (
            scene_label == "enemy"
            and scene_confidence >= 0.88
        )

        if not enemy_seen:
            return action

        ammo = int(ammo or 0)
        health = int(health or 0)

        # Never choose shoot with no ammo.
        if ammo <= 0:
            if health <= 50:
                return "move_backward"
            return "strafe_right"

        # Low health means survival first.
        if health <= 40:
            if self._step_count % 3 == 0:
                return "move_backward"
            return "strafe_right"

        # Controlled burst pattern:
        # 2 shoot steps, then movement/spacing.
        cycle = self._step_count % 8

        if cycle in [0, 1]:
            return "shoot"

        if cycle in [2, 3]:
            return "strafe_left"

        if cycle in [4, 5]:
            return "move_backward"

        return action


    def sanitize_action(self, action, game_state, enemy_visible):
        """
        Final action gate for the current curriculum stage.

        This blocks illegal/impossible actions, but it does NOT create new
        combat behavior. In particular, it must not convert swap_weapon or
        no-ammo cases back into shoot.
        """
        config = self.get_stage_config()

        ammo = int(game_state.get("ammo", 0) or 0)
        current_weapon = str(game_state.get("weapon", "")).lower()

        is_melee_weapon = (
            "fist" in current_weapon
            or "chainsaw" in current_weapon
            or "ripter" in current_weapon
            or "melee" in current_weapon
        )

        if action == "use" and not config.get("allow_use", False):
            return "move_forward"

        if action == "shoot":
            if not config.get("allow_shoot", False):
                return "move_forward"

            # No ammo means no shoot. Do not convert back into shoot later.
            if ammo <= 0:
                if is_melee_weapon and enemy_visible and config.get("allow_melee", False):
                    return "melee_attack"
                return "move_backward"

            if config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
                return "move_forward"

            return action

        if action == "melee_attack":
            if not config.get("allow_melee", False):
                return "move_forward"

            if not enemy_visible:
                return "move_forward"

            return action

        if action == "swap_weapon":
            # Swap is allowed, but not repeatedly.
            # Never turn swap_weapon into shoot here.
            if self.consecutive_swap_steps >= 2:
                return "move_backward" if enemy_visible else "move_forward"
            return action

        return action

    def _reset_stage_counters(self):
        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.key_item_count = 0
        self.enemy_engagement_count = 0
        self.track_enemy_count = 0
        self.dodge_enemies_count = 0
        self.valid_shot_count = 0
        self.enemy_visible_steps = 0
        self.swap_weapon_count = 0
        self.stuck_counter = 0
        self.distance_traveled = 0.0
        self.visited_tiles.clear()
        self.visited_areas.clear()
        self.melee_attack_count = 0
        self.melee_close_bonus_count = 0
        self.weapon_pickup_count = 0
        self.ammo_pickup_count = 0
        self.enemy_kill_count = 0
        self.combat_survival_steps = 0
        self.dodge_when_damaged_count = 0
        self.retreat_from_close_enemy_count = 0
        self.wall_contact_steps = 0
        self.last_wall_escape_step = -100
        self.route_progress_level = 0
        self.corner_trap_position = None
        self.corner_trap_steps = 0
        self.last_corner_escape_step = -100

        self.consecutive_melee_steps = 0
        self.last_melee_step = -100
        self.consecutive_swap_steps = 0
        self.last_swap_step = -100

        self.wall_escape_mode = False
        self.wall_escape_step = 0
        self.wall_escape_direction = "right"
        self.wall_escape_start_step = -100

    def close(self):
        self.controller.release_all()

        if self.record:
            if self.recorder is not None:
                self.recorder.save()

            if self.event_recorder is not None:
                self.event_recorder.save()

        if self._kb_listener is not None:
            self._kb_listener.stop()

        if self.game_process is not None:
            self.game_process.terminate()
            self.game_process = None
