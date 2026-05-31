import sys
import os
import cv2
import json
import math
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

try:
    from pynput import keyboard
except Exception as e:
    print(f"[keyboard] pynput disabled: {e}")
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
try:
    from recording.record_player_session import PlayerRecorder
except Exception as e:
    print(f"[recording] PlayerRecorder disabled: {e}")
    PlayerRecorder = None

try:
    from recording.event_recorder import EventRecorder
except Exception as e:
    print(f"[recording] EventRecorder disabled: {e}")
    EventRecorder = None
from curriculum.curriculum_manager import CurriculumManager
from observation.frame_processor import FrameProcessor
from navigation.checkpoint_tracker import CheckpointTracker
from navigation.level_guides import get_level_guide
from vision.scene_predictor import ScenePredictor
from sensory.sensory_model import SensoryModel
from director.route_director import RouteDirector
from navigation.retrace_navigator import RetraceNavigator
from observation.wall_sensor import WallSensor
from env.shared_doom_logic import SharedDoomLogic
from perception.prediction_adapter import build_doomretro_game_state
from imitation.action_prior_advisor import ActionPriorAdvisor

try:
    from observation.vision_detector import VisionDetector
except ImportError:
    VisionDetector = None

try:
    from vision.object_predictor import ObjectPredictor
except Exception:
    ObjectPredictor = None

try:
    from vision.hud_predictor import HudPredictor
except Exception as e:
    print(f"[hud_vision] HudPredictor disabled: {e}")
    HudPredictor = None

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

    def __init__(self, launch_doom=True, record=False, training_mode=True):
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
        self.wall_sensor = WallSensor()
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
        self.sensory_model = SensoryModel()
        self.route_director = RouteDirector()
        self.retrace_navigator = RetraceNavigator()
        self.enable_retrace_navigator = False
        self.shared_logic = SharedDoomLogic()
        self.previous_shared_game_state = None
        self.last_shared_debug = {}
        self.use_shared_logic_reward = True
        self.use_shared_logic_reward_blend = True
        self.shared_logic_reward_scale = 0.10
        self.shared_logic_reward_clip = 0.25
        self.use_shared_movement_reward = True
        self.shared_movement_reward_scale = 1.0
        self.use_shared_wall_reward = True
        self.shared_wall_reward_scale = 1.0
        self.use_shared_route_curriculum_reward = True
        self.shared_route_curriculum_reward_scale = 1.0
        self.use_shared_object_scene_reward = True
        self.shared_object_scene_reward_scale = 1.0
        self.retrace_lock_steps = 0
        self.pending_retrace_action = None
        self.secret_area_reached_this_episode = False
        self.secrets_found_this_episode = set()
        self.training_mode = training_mode
        self.last_progress_position = None
        self.no_position_change_steps = 0
        self.last_distance_to_goal = None
        self.no_distance_progress_steps = 0
        # in __init__
        self.recent_loop_tile = None
        self.recent_loop_steps = 0

        # Full replacement mode exists, but keep it OFF until tests pass.
        self.use_full_shared_reward_mode = True
        self.full_shared_reward_scale = 1.0
        self.full_shared_reward_clip = 5.0

        self.old_route_reward_scale = 0.25
        self.old_curriculum_reward_scale = 0.25

        self.old_wall_reward_scale = 0.25
        self.old_stuck_reward_scale = 0.25
        self.old_object_reward_scale = 0.25
        self.old_scene_reward_scale = 0.25
        self.old_combat_reward_scale = 0.25
        
        # -----------------------------------------------------
        # Helper control switches
        # -----------------------------------------------------
        # Keep these False while PPO is learning.
        # These systems should guide with reward/logging, not hijack actions.
        self.enable_sensory_action_override =False
        self.enable_goal_assist_action_override = False

        # Keep wall safety on, but only for true front-wall emergencies.
        self.enable_vision_blocker_override = False

        # -----------------------------------------------------
        # Death review buffer
        # -----------------------------------------------------
        # Stores recent frames so the agent can save what happened before death.
        self.recent_frame_buffer = deque(maxlen=60)
        self.death_capture_dir = Path("vision_dataset_v2/death_review")
        self.death_capture_dir.mkdir(parents=True, exist_ok=True)
        self.death_capture_count = 0
        self.episode_had_death = False
        self.episode_had_stuck_reset = False

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
        self.scene_predictor = None
        self.use_scene_classifier = True
        # When sensory emergency is active, it becomes the single recovery driver.
        self.sensory_emergency_active = False
        self.sensory_emergency_steps = 0

        if VisionDetector is not None:
            self.vision_detector = VisionDetector(frame_processor=self.frame_processor)
        else:
            self.vision_detector = None
            print("[vision] observation.vision_detector not found — using built-in fallback vision.")

        self.checkpoint_tracker = CheckpointTracker(reach_radius=128.0)

        self.current_level_name = "freedoom1_e1m1"
        self.level_guide = get_level_guide(self.current_level_name)

        self.checkpoint_tracker.set_level_guide(
            checkpoints=self.level_guide.get("checkpoints", []),
            secrets=self.level_guide.get("secrets", []),
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
        self.event_recorder = EventRecorder() if EventRecorder is not None else None
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
            self.recorder = PlayerRecorder() if PlayerRecorder is not None else None
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
        if self.training_mode:
            self.collect_vision_frames = False
            self.enable_smart_clips = False
            self.enable_death_review_capture = False
        else:
            self.enable_smart_clips = True
            self.enable_death_review_capture = True

        self.action_space = spaces.Discrete(len(self.actions))
        self.use_action_prior_advice = True
        self.action_prior_reward_scale = 0.005
        self.action_prior_min_confidence = 0.70

        action_prior_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "checkpoints",
            "action_prior_from_teacher_balanced.pt",
        )

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
            print(f"[action_prior] disabled: {e}")
            self.action_prior_advisor = None
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8,
        )

        # -----------------------------------------------------
        # Curriculum
        # -----------------------------------------------------

        self.curriculum_stage = 0
        self.max_stage = 7
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

        # -----------------------------------------------------
        # First-level route progress tracking
        # -----------------------------------------------------
        self.route_zones_reached = set()
        self.best_route_progress_level = 0

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
        self.current_weapon_name = "pistol"
        self.has_berserk = False
        self.berserk_steps_remaining = 0
        self.last_weapon_swap_step = -100
        self.last_forced_weapon_reason = None
        self.weapon_swap_cooldown = 25

        self.consecutive_melee_steps = 0
        self.last_melee_step = -100
        self.consecutive_swap_steps = 0
        self.last_swap_step = -100
        # -----------------------------------------------------
        # Smart weapon selection memory
        # -----------------------------------------------------
        self.last_weapon_select_step = -100
        self.weapon_select_cooldown = 18
        self.last_weapon_key = None

        # -----------------------------------------------------
        # Behavior counters
        # -----------------------------------------------------

        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.last_use_step = -100
        self.last_use_position = None
        self.pending_use_check = None
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

        if ObjectPredictor is not None:
            try:
                self.object_predictor = ObjectPredictor()
                print("[object_vision] Loaded object_multilabel_classifier.pt")
            except Exception as e:
                print(f"[object_vision] Could not load object predictor: {e}")
                self.object_predictor = None

        if HudPredictor is not None:
            try:
                self.hud_predictor = HudPredictor()
                print("[hud_vision] Loaded lightweight HUD predictor")
            except Exception as e:
                print(f"[hud_vision] Could not initialize HUD predictor: {e}")
                self.hud_predictor = None
        else:
            self.hud_predictor = None

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
                "name": "complete_level_basic",
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
                "allow_use": True,
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
                "move_backward",
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
    def hud_status_reward(self, hud_result, action):
        """
        Small HUD-aware reward shaping.

        This teaches the agent that low health and low ammo are important,
        without letting HUD logic directly control the action.
        """

        if not hud_result:
            return 0.0

        reward = 0.0

        low_health = hud_result.get("low_health", False)
        low_ammo = hud_result.get("low_ammo", False)
        damage_flash = hud_result.get("damage_flash", False)

        if low_health:
            reward += self.add_penalty("hud_low_health", -0.05)

        if low_ammo and action == "shoot":
            reward += self.add_penalty("hud_low_ammo_shoot", -0.05)

        if damage_flash:
            reward += self.add_penalty("hud_damage_flash", -0.03)


        return reward

    def door_use_reward(self, action, game_state, scene_label, scene_confidence):
        """
        Rewards useful 'use' actions and penalizes use spam.

        Useful use means:
        - use near a door/button/elevator-looking area
        - use followed by position/progress change
        """

        reward = 0.0

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        pos = np.array([float(x), float(y)], dtype=np.float32)

        door_like = (
            scene_label == "door_or_button"
            and scene_confidence >= 0.35
        )

        if action == "use":
            self.door_interaction_count += 1

            if door_like:
                reward += 0.10
                self.reward_manager.add("use_near_door_like_scene", 0.10)
            else:
                reward += self.add_penalty("use_not_near_door", -0.05)

            if self.last_use_step >= 0 and self._step_count - self.last_use_step < 10:
                reward += self.add_penalty("use_spam", -0.10)

            self.last_use_step = self._step_count
            self.last_use_position = pos.copy()
            self.pending_use_check = {
                "step": self._step_count,
                "position": pos.copy(),
                "route_progress": self.route_progress_level,
            }

        if self.pending_use_check is not None:
            age = self._step_count - self.pending_use_check["step"]

            if 2 <= age <= 20:
                old_pos = self.pending_use_check["position"]
                moved = float(np.linalg.norm(pos - old_pos))
                route_improved = (
                    self.route_progress_level
                    > self.pending_use_check["route_progress"]
                )

                if moved >= 32.0 or route_improved:
                    door_or_route_context = (
                        route_improved
                        or scene_label in ["door_or_button", "door", "switch", "elevator", "lift"]
                        or game_state.get("sensory_situation") in ["door_possible", "right_route_area"]
                    )

                    if door_or_route_context and (moved >= 32.0 or route_improved):
                        reward += 0.40
                        self.reward_manager.add("use_created_progress", 0.40)
                        print(
                            f"[door_use] useful_use moved={moved:.1f} "
                            f"route_improved={route_improved}"
                        )
                        self.pending_use_check = None

        return reward
    

    def real_map_navigation_reward(self, game_state):
        """
        Real-coordinate navigation shaping.

        This rewards:
        - getting closer to the actual exit
        - staying near the real exit lane
        - reaching real WAD-based waypoint bubbles

        It penalizes:
        - drifting far east/right into the old fake route area
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        guide = getattr(self, "level_guide", {}) or {}
        goal = guide.get("main_goal") or {}

        gx = goal.get("x")
        gy = goal.get("y")

        if gx is None or gy is None:
            return 0.0

        gx = float(gx)
        gy = float(gy)

        dx = x - gx
        dy = y - gy
        dist = (dx * dx + dy * dy) ** 0.5

        reward = 0.0

        # Main progress reward.
        previous = getattr(self, "best_main_goal_distance", None)

        if previous is None:
            self.best_main_goal_distance = dist
        else:
            improvement = previous - dist

            if improvement > 0:
                reward += min(0.10, improvement * 0.003)
                self.best_main_goal_distance = min(previous, dist)
            elif improvement < -24:
                reward -= 0.03

        # Soft lane shaping: exit is around x=-400.
        lateral_error = abs(x - gx)

        if lateral_error <= 192:
            reward += 0.01
        elif lateral_error > 600:
            reward -= 0.04
        elif lateral_error > 384:
            reward -= 0.02

        # Penalize extreme east drift, which the old route caused.
        bounds = guide.get("safe_bounds") or {}
        max_x = float(bounds.get("max_x", 200.0))
        min_x = float(bounds.get("min_x", -900.0))

        if x > max_x:
            reward += float(bounds.get("soft_penalty", -0.02))
            if x > max_x + 500:
                reward += float(bounds.get("hard_penalty", -0.08))

        if x < min_x:
            reward += float(bounds.get("soft_penalty", -0.02))

        return float(reward)


    def route_progress_reward(self, game_state):
        """
        Real-map route progress.

        This uses route_zones from navigation/level_guides.py.
        No hardcoded old route names are allowed here.
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        route_zones = self.level_guide.get("route_zones", [])

        if not route_zones:
            return 0.0

        reward = 0.0

        for idx, zone in enumerate(route_zones):
            name = zone.get("name", f"route_zone_{idx}")
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            radius = float(zone.get("radius", 128.0))
            zone_reward = float(zone.get("reward", 0.05))

            if name in self.route_zones_reached:
                continue

            dx = x - zx
            dy = y - zy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= radius:
                self.route_zones_reached.add(name)
                self.route_progress_level = max(self.route_progress_level, idx + 1)
                reward += zone_reward

                print(
                    f"[route_progress] reached={name} "
                    f"level={self.route_progress_level} "
                    f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
                )

        return float(reward)


    def tile_exploration_reward(self, game_state):
        """
        Simple exploration reward using player position tiles.

        This is easier to debug than RND and helps stop same-area loops.
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        tile_size = 64
        tile = (int(float(x) // tile_size), int(float(y) // tile_size))

        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            self.reward_manager.add("new_tile", 0.03)
            return 0.03

        if self.repeated_position_steps >= 10:
            self.reward_manager.add("same_tile_loop_penalty", -0.03)
            return self.add_penalty("same_tile_loop_penalty", -0.03)

        return 0.0
    
    def update_powerup_state(self, game_state, object_result=None):
        """
        Track powerups using object labels and game-state hints.

        Berserk in Doom/Freedoom may appear as a black/dark medkit-like pickup.
        We intentionally use aliases because Freedoom sprite/model labels may differ.
        """

        berserk_aliases = {
            "berserk",
            "berserk_pack",
            "berserker",
            "black_medkit",
            "dark_medkit",
            "powerup_berserk",
            "strength_powerup",
        }

        labels = set()

        if object_result:
            labels.update(object_result.get("present", []))

            scores = object_result.get("scores", {})
            for label, score in scores.items():
                try:
                    if float(score) >= 0.50:
                        labels.add(label)
                except Exception:
                    pass

        if labels.intersection(berserk_aliases):
            self.has_berserk = True
            # Treat as level-long for training, but keep a debug counter.
            self.berserk_steps_remaining = max(self.berserk_steps_remaining, 5000)
            self.last_forced_weapon_reason = "berserk_pickup"

        if self.berserk_steps_remaining > 0:
            self.berserk_steps_remaining -= 1

        game_state["has_berserk"] = self.has_berserk
        game_state["berserk_steps_remaining"] = self.berserk_steps_remaining

        return game_state


    def infer_enemy_distance_tiles(self, game_state, object_result=None):
        """
        Estimate enemy distance in tiles.

        This is intentionally conservative. If no distance source exists,
        return None instead of guessing.
        """

        if game_state.get("enemy_distance_tiles") is not None:
            return game_state.get("enemy_distance_tiles")

        if game_state.get("enemy_distance") is not None:
            try:
                return float(game_state["enemy_distance"]) / 64.0
            except Exception:
                return None

        if object_result:
            if object_result.get("enemy_distance_tiles") is not None:
                return object_result.get("enemy_distance_tiles")

            if object_result.get("enemy_distance") is not None:
                try:
                    return float(object_result["enemy_distance"]) / 64.0
                except Exception:
                    return None

        return None


    def infer_enemy_count(self, game_state, object_result=None):
        """
        Estimate visible enemy count.

        If the object model cannot count enemies yet, fall back to 1 when enemy_visible is true.
        """

        if game_state.get("enemy_count") is not None:
            try:
                return int(game_state["enemy_count"])
            except Exception:
                pass

        if object_result:
            if object_result.get("enemy_count") is not None:
                try:
                    return int(object_result["enemy_count"])
                except Exception:
                    pass

            present = object_result.get("present", [])
            if "enemy_visible" in present:
                return 1

        if game_state.get("enemy_visible", False):
            return 1

        return 0


    def tactical_weapon_action(
        self,
        action,
        game_state,
        object_result=None,
        wall_info=None,
    ):
        """
        Situational weapon swapping.

        This prevents blind swap_weapon spam while still allowing:
        - berserk/fist for very close enemies
        - guns for medium/far enemies
        - rockets only for groups at safe distance
        """

        if self.curriculum_stage < 3:
            if action == "swap_weapon":
                self.last_forced_weapon_reason = "swap_blocked_before_combat_stage"
                return "move_forward"
            return action

        if action != "swap_weapon":
            return action

        if self._step_count - self.last_weapon_swap_step < self.weapon_swap_cooldown:
            self.last_forced_weapon_reason = "swap_cooldown"
            return "move_backward"

        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        enemy_distance_tiles = self.infer_enemy_distance_tiles(game_state, object_result)
        enemy_count = self.infer_enemy_count(game_state, object_result)

        current_weapon = str(
            game_state.get("weapon", self.current_weapon_name)
        ).lower()

        front_ratio = 0.0
        if wall_info:
            front_ratio = float(wall_info.get("front_ratio", 0.0))

        rocket_unsafe = (
            front_ratio >= 0.45
            or (
                enemy_distance_tiles is not None
                and enemy_distance_tiles < 4.0
            )
        )

        if not enemy_visible:
            self.last_forced_weapon_reason = "swap_blocked_no_enemy"
            return "move_forward"

        if self.has_berserk and enemy_distance_tiles is not None:
            if enemy_distance_tiles <= 1.5 and enemy_centered:
                self.last_weapon_swap_step = self._step_count
                self.last_forced_weapon_reason = "berserk_close_enemy"
                return "swap_weapon"

            if current_weapon in ["fist", "chainsaw", "riptor"] and enemy_distance_tiles > 2.0:
                self.last_weapon_swap_step = self._step_count
                self.last_forced_weapon_reason = "leave_melee_range"
                return "swap_weapon"

        if enemy_count >= 2 and not rocket_unsafe:
            self.last_weapon_swap_step = self._step_count
            self.last_forced_weapon_reason = "group_enemy_safe_rocket"
            return "swap_weapon"

        ammo = int(game_state.get("ammo", 0) or 0)

        self.last_forced_weapon_reason = "swap_not_contextual"

        if ammo <= 0:
            return "move_backward"

        return "shoot" if enemy_centered else "move_backward"

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
        self.route_zones_reached = set()
        self.best_route_progress_level = 0
        self.goal_bubble_best_dist = {}
        self.goal_bubble_last_dist = {}
        self.goal_bubble_stall_steps = {}
        self.goal_turn_steps = 0
        self.recent_position_tiles = []
        self.repeated_position_steps = 0
        self.last_sensory_scene_label = "unclear"
        self.last_sensory_scene_confidence = 0.0
        self.secret_area_reached_this_episode = False
        self.secrets_found_this_episode = set()
        self.last_sensory_scene_label = "unclear"
        self.last_sensory_scene_confidence = 0.0
        self.sensory_model.reset()
        self.route_director.reset()
        self.retrace_navigator.reset()
        self.retrace_lock_steps = 0
        self.pending_retrace_action = None
        self.recent_loop_tile = None
        self.recent_loop_steps = 0
        self.last_progress_position = None
        self.no_position_change_steps = 0
        self.last_distance_to_goal = None
        self.no_distance_progress_steps = 0
        self.last_exit_distance = None
        self.best_exit_distance = None
        self.best_main_goal_distance = None

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
        self.current_weapon_name = "pistol"
        self.has_berserk = False
        self.berserk_steps_remaining = 0
        self.last_weapon_swap_step = -100
        self.last_forced_weapon_reason = None

        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.last_use_step = -100
        self.last_use_position = None
        self.pending_use_check = None
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
        self.shared_logic.reset_episode()
        self.previous_shared_game_state = None
        self.last_shared_debug = {}

        self.reward_manager.reset()
        self.checkpoint_tracker.reset()
        self.corridor_reached_this_episode = False
        self.episode_had_death = False
        self.episode_had_stuck_reset = False
        self.recent_frame_buffer.clear()

        # Do not clear self.secret_use_locations every episode.
        # This memory should persist across episodes.
    

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

        scene_label = "unclear"
        scene_confidence = 0.0
        scene_probs = {}

        object_result = None
        hud_result = None
        pre_hud_result = None
        info = {
            "shared_logic_debug": self.last_shared_debug,
            "use_shared_logic_reward": self.use_shared_logic_reward,
            }

        safe_frame = self.observer.build()
        safe_observation = self.frame_stack.add_frame(safe_frame)

        if action_index not in self.get_valid_actions():
            reward -= 0.2
            self.episode_reward_total += float(reward)
            return safe_observation, float(reward), terminated, truncated, info

        action = self.actions[action_index]

        if self.retrace_lock_steps > 0:
            self.retrace_lock_steps -= 1

        can_use_exploration_override = (
            self.retrace_lock_steps <= 0
            and self.stuck_counter < 4
            and self.wall_contact_steps < 2
            and not getattr(self.retrace_navigator, "active", False)
            and not self.sensory_emergency_active
        )

        object_result = None
        hud_result = None

        decision_trace = {
            "ppo_action": action,
            "final_action": None,
            "scene_label": None,
            "scene_confidence": None,
            "x": None,
            "y": None,
            "stuck_counter": self.stuck_counter,
            "wall_contact_steps": self.wall_contact_steps,
            "changes": [],
        }
        config = self.get_stage_config()

        # -----------------------------------------------------
        # Pre-action perception
        # -----------------------------------------------------

        pre_frame = self.observer.get_frame()

        pre_gameplay_frame = self.frame_processor.make_model_frame(
            pre_frame,
            view_mode="gameplay_wide",
            size=(160, 100),
        )

        pre_center_frame = self.frame_processor.make_model_frame(
            pre_frame,
            view_mode="gameplay_center",
            size=(84, 84),
        )

        pre_hud_frame = self.frame_processor.make_model_frame(
            pre_frame,
            view_mode="hud",
            size=(160, 32),
        )

        pre_game_state = self.observer.get_game_state()
        pre_vision = self.detect_vision(pre_frame)

        wall_sensor_state = self.wall_sensor.analyze(pre_frame)

        pre_wall_info = self._wall_direction_info(pre_frame)

        pre_wall_info["left_ratio"] = max(
            pre_wall_info.get("left_ratio", 0.0),
            wall_sensor_state["left_ratio"],
        )
        pre_wall_info["front_ratio"] = max(
            pre_wall_info.get("front_ratio", 0.0),
            wall_sensor_state["front_ratio"],
        )
        pre_wall_info["right_ratio"] = max(
            pre_wall_info.get("right_ratio", 0.0),
            wall_sensor_state["right_ratio"],
        )
        pre_wall_info["front_wall"] = (
            pre_wall_info.get("front_wall", False)
            or wall_sensor_state["front_blocked"]
        )

        pre_game_state["wall_sensor"] = wall_sensor_state
        pre_game_state["left_wall_ratio"] = wall_sensor_state["left_ratio"]
        pre_game_state["front_wall_ratio"] = wall_sensor_state["front_ratio"]
        pre_game_state["right_wall_ratio"] = wall_sensor_state["right_ratio"]
        pre_game_state["front_blocked"] = wall_sensor_state["front_blocked"]

        # -----------------------------------------------------
        # Pre-action object vision
        # -----------------------------------------------------
        # Used for smart weapon selection before Doom receives input.
        pre_object_result = None

        if self.object_predictor is not None and pre_frame is not None:
            try:
                pre_object_result = self.object_predictor.predict(pre_frame)
                pre_game_state["object_vision"] = pre_object_result

                if self._step_count % 10 == 0:
                    print(
                        "[pre_object_vision] "
                        f"present={pre_object_result.get('present', [])} "
                        f"enemy={pre_object_result.get('scores', {}).get('enemy_visible', 0.0):.2f} "
                        f"barrel={pre_object_result.get('scores', {}).get('explosive_barrel', 0.0):.2f}"
                    )

            except Exception as e:
                print(f"[pre_object_vision] prediction failed: {e}")
                pre_object_result = None

        pre_scene_result = None

        if self.scene_predictor is not None:
            pre_scene_result = self.scene_predictor.predict(
                pre_gameplay_frame,
                view_mode="gameplay_wide",
            )

        pre_object_result = None

        if self.object_predictor is not None:
            pre_object_result = self.object_predictor.predict(
                pre_gameplay_frame,
                view_mode="gameplay_wide",
            )

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
            pre_scene_result = self.scene_predictor.predict(pre_frame, view_mode="wide")
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
        # Pending retrace action
        # -----------------------------------------------------
        # Retrace is computed after the previous action, so we apply it
        # at the beginning of the next step before exploration can override it.
        if (
            self.pending_retrace_action is not None
            and self.pending_retrace_action in self.get_allowed_actions()
        ):
            pending = self.pending_retrace_action

            front_blocked = (
                pre_wall_info.get("front_wall", False)
                or pre_wall_info.get("front_ratio", 0.0) >= 0.55
            )

            if pending == "move_forward" and front_blocked:
                before_pending = pending

                if pre_wall_info.get("left_ratio", 0.0) < pre_wall_info.get("right_ratio", 0.0):
                    pending = "turn_left"
                else:
                    pending = "turn_right"

                print(
                    f"[override] pending_retrace_safety: "
                    f"{before_pending} -> {pending} "
                    f"front={pre_wall_info.get('front_ratio', 0.0):.2f}"
                )

            before = action
            action = pending
            pre_game_state["action"] = action
            self.pending_retrace_action = None
            self.retrace_lock_steps = max(self.retrace_lock_steps, 6)
            can_use_exploration_override = False

            print(
                f"[override] pending_retrace: {before} -> {action} "
                f"lock={self.retrace_lock_steps}"
            )

        pre_game_state = self.update_powerup_state(
            game_state=pre_game_state,
            object_result=pre_object_result,
        )

        # -----------------------------------------------------
        # Pre-action sensory override
        # -----------------------------------------------------
        # The neural classifier can call spawn-boundary/corner views open_path.
        # The sensory layer corrects that using position memory and wall ratios.
        
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

        pre_wall_info["left_ratio"] = max(
            pre_wall_info.get("left_ratio", 0.0),
            wall_sensor_state["left_ratio"],
        )
        pre_wall_info["front_ratio"] = max(
            pre_wall_info.get("front_ratio", 0.0),
            wall_sensor_state["front_ratio"],
        )
        pre_wall_info["right_ratio"] = max(
            pre_wall_info.get("right_ratio", 0.0),
            wall_sensor_state["right_ratio"],
        )
        pre_wall_info["front_wall"] = (
            pre_wall_info.get("front_wall", False)
            or wall_sensor_state["front_blocked"]
        )

        wall_penalty = self.strong_wall_penalty(
            action=action,
            wall_info=pre_wall_info,
            game_state=pre_game_state,
        )

        reward += wall_penalty

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
            if False and can_use_exploration_override:
                before = action
                action = self.exploration_assist_action(
                    action=action,
                    enemy_visible=False,
                    distance_moved=None,
                    motion=None,
                )
                if action != before:
                    print(f"[override] exploration: {before} -> {action}")
                    decision_trace["changes"].append(("exploration", before, action))
            else:
                if self.retrace_lock_steps > 0 or getattr(self.retrace_navigator, "active", False):
                    print(
                        f"[override_blocked] exploration blocked "
                        f"lock={self.retrace_lock_steps} "
                        f"retrace_active={getattr(self.retrace_navigator, 'active', False)} "
                        f"stuck={self.stuck_counter} "
                        f"wall={self.wall_contact_steps}"
                    )

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

            allow_aim_priority = (
                pre_enemy_visible
                and not pre_enemy_centered
                and self.retrace_lock_steps <= 0
                and not getattr(self.retrace_navigator, "active", False)
                and self.stuck_counter < 4
            )

            if allow_aim_priority:
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
            else:
                if pre_enemy_visible and not pre_enemy_centered and self._step_count % 25 == 0:
                    print(
                        f"[override_blocked] aim_priority blocked "
                        f"lock={self.retrace_lock_steps} "
                        f"retrace_active={getattr(self.retrace_navigator, 'active', False)} "
                        f"stuck={self.stuck_counter}"
                    )

            if not aimed_this_step:
                before = action
                pre_game_state["action"] = action
                tactical_action = None   #self.combat_tactics.choose_combat_action(pre_game_state)

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
                    ammo = int(pre_game_state.get("ammo", 0) or 0)

                    allow_aim_assist = (
                        pre_enemy_visible
                        and not pre_enemy_centered
                        and ammo > 0
                        and self.retrace_lock_steps <= 0
                        and not getattr(self.retrace_navigator, "active", False)
                        and self.stuck_counter < 4
                    )

                    if allow_aim_assist:
                        before = action
                        action = self.aim_assist_action(
                            action=action,
                            frame=pre_frame,
                            enemy_visible=pre_enemy_visible,
                            enemy_centered=pre_enemy_centered,
                            ammo=ammo,
                        )
                        if action != before:
                            print(f"[override] aim_assist: {before} -> {action}")
                    else:
                        if pre_enemy_visible and self._step_count % 25 == 0:
                            print(
                                f"[override_blocked] aim_assist blocked "
                                f"centered={pre_enemy_centered} ammo={ammo} "
                                f"lock={self.retrace_lock_steps} "
                                f"retrace_active={getattr(self.retrace_navigator, 'active', False)} "
                                f"stuck={self.stuck_counter}"
                            )

            if False and can_use_exploration_override:
                before = action
                action = self.exploration_assist_action(
                    action=action,
                    enemy_visible=pre_enemy_visible,
                    distance_moved=None,
                    motion=None,
                )
                if action != before:
                    print(f"[override] exploration: {before} -> {action}")
            else:
                if self.retrace_lock_steps > 0 or getattr(self.retrace_navigator, "active", False):
                    print(
                        f"[override_blocked] exploration blocked "
                        f"lock={self.retrace_lock_steps} "
                        f"retrace_active={getattr(self.retrace_navigator, 'active', False)} "
                        f"stuck={self.stuck_counter} "
                        f"wall={self.wall_contact_steps}"
                    )

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
        if self.curriculum_stage < 3:
            if pre_object_result is not None:
                pre_object_result["present"] = [
                    label for label in pre_object_result.get("present", [])
                    if label != "enemy_visible"
                ]
                pre_object_result["scores"]["enemy_visible"] = 0.0

        # -----------------------------------------------------
        # Goal assist should NOT hijack PPO actions.
        # The director can still provide reward/hints later.
        # -----------------------------------------------------
        if self.enable_goal_assist_action_override:
            before_goal = action
            action = self.goal_assist_action(
                action=action,
                game_state=pre_game_state,
                wall_info=pre_wall_info,
                enemy_visible=pre_enemy_visible,
            )
            if action != before_goal:
                print(f"[override] goal_assist: {before_goal} -> {action}")

        door_priority = (
            pre_scene_label == "door_or_button"
            and pre_scene_confidence >= 0.35
            and self.curriculum_stage >= 2
        )

        if door_priority and action in ["move_forward", "move_backward", "turn_left", "turn_right"]:
            before = action
            action = "use"
            print(
                f"[override] door_priority: {before} -> use "
                f"label={pre_scene_label} conf={pre_scene_confidence:.2f}"
            )

        true_front_blocker = (
            pre_scene_label in ["front_wall", "obstacle"]
            and pre_scene_confidence >= 0.85
        )

        repeated_wall_contact = (
            self.wall_contact_steps >= 3
            or self.stuck_counter >= 8
        )

        if (
            self.enable_vision_blocker_override
            and action == "move_forward"
            and (true_front_blocker or repeated_wall_contact)
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
            elif pre_scene_label == "boundary_or_stuck_wall" and repeated_wall_contact:
                # Spawn boundaries and corner loops need a stronger escape cycle.
                if self._step_count % 6 in [0, 1]:
                    action = "move_backward"
                elif self._step_count % 6 in [2, 3]:
                    action = "turn_right"
                else:
                    action = "strafe_right"
            else:
                # Normal front walls should rotate away first, not back up forever.
                if self._step_count % 3 == 0:
                    action = "turn_right"
                elif self._step_count % 3 == 1:
                    action = "turn_left"
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

        if (
            not self.sensory_emergency_active
            and self.retrace_lock_steps <= 0
            and not getattr(self.retrace_navigator, "active", False)
        ):
            action = self.wall_bubble_action(action, pre_wall_bubble)
        else:
            if self.retrace_lock_steps > 0 or getattr(self.retrace_navigator, "active", False):
                print(
                    f"[override_blocked] wall_bubble blocked "
                    f"lock={self.retrace_lock_steps} "
                    f"retrace_ctive={getattr(self.retrace_navigator, 'active', False)}"
                )

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

        if self.stuck_counter >= 8 and not self.sensory_emergency_active:
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
            allowed = self.get_allowed_actions()

            if "move_backward" in allowed and (
                pre_wall_info.get("front_wall", False)
                or pre_wall_info.get("front_ratio", 0.0) > 0.35
                or self.stuck_counter >= 2
                or self.wall_contact_steps >= 2
            ):
                fixed_action = "move_backward"

            elif "turn_left" in allowed and "turn_right" in allowed:
                left_ratio = float(pre_wall_info.get("left_ratio", 0.0) or 0.0)
                right_ratio = float(pre_wall_info.get("right_ratio", 0.0) or 0.0)
                fixed_action = "turn_left" if left_ratio <= right_ratio else "turn_right"

            else:
                fixed_action = allowed[0]

            print(
                f"[override] final_safety: {action} -> {fixed_action} "
                f"because stage={self.curriculum_stage} allowed={allowed}"
            )

            action = fixed_action

        helper_action = self.helper.get_action(pre_game_state)

        if action == helper_action:
            reward += 0.05
        else:
            reward -= 0.01

        decision_trace["final_action"] = action
        decision_trace["scene_label"] = pre_scene_label
        decision_trace["scene_confidence"] = pre_scene_confidence
        decision_trace["x"] = pre_game_state.get("x")
        decision_trace["y"] = pre_game_state.get("y")

        if self._step_count % 25 == 0:
            print("[decision_trace]", decision_trace)

        # -----------------------------------------------------
        # Execute action
        # -----------------------------------------------------

        before_weapon_guard = action
        action = self.tactical_weapon_action(
            action=action,
            game_state=pre_game_state,
            object_result=pre_object_result,
            wall_info=pre_wall_info,
        )
        if action == "shoot" and int(pre_game_state.get("ammo", 0) or 0) <= 0:
            before = action
            action = "move_backward"
            print(f"[override] no_ammo_after_tactical: {before} -> {action}")

        if action != before_weapon_guard:
            print(
                f"[override] tactical_weapon: {before_weapon_guard} -> {action} "
                f"reason={self.last_forced_weapon_reason} "
                f"berserk={self.has_berserk}"
            )

        action_prior_reward, action_prior_result = self.action_prior_advice_reward(
            frame=pre_frame,
            action=action,
        )

        reward += action_prior_reward

        if action_prior_result is not None:
            info["action_prior"] = action_prior_result
            self.last_shared_debug["action_prior"] = action_prior_result
            self.last_shared_debug["action_prior_reward"] = action_prior_reward

        if action == "swap_weapon":
            selected = self.smart_weapon_select(
                game_state=pre_game_state,
                object_result=pre_object_result,
                enemy_visible=pre_enemy_visible,
                enemy_centered=pre_enemy_centered,
            )

            if not selected:
                self.perform_action(action)
        else:
            self.perform_action(action)

        if hasattr(self.controller, "last_action_sent"):
            info["action_sent"] = self.controller.last_action_sent

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

        try:
            shared_game_state = self._build_shared_game_state(
                base_game_state=game_state,
                object_result=object_result or pre_object_result,
                scene_result={
                    "label": scene_label,
                    "confidence": scene_confidence,
                },
                hud_result=locals().get("hud_result") or locals().get("pre_hud_result"),
            )

            shared_reward, shared_debug = self.shared_logic.compute_reward(
                previous_state=self.previous_shared_game_state,
                current_state=shared_game_state,
                action_name=str(action),
                depth_obs=None,
            )

            self.previous_shared_game_state = shared_game_state
            self.last_shared_debug = shared_debug
            self.last_shared_debug["shared_reward_preview"] = shared_reward

        except Exception as e:
            self.last_shared_debug = {
                "shared_logic_error": str(e),
            }


        reward += self.route_progress_reward(game_state)
        reward += self.real_map_navigation_reward(game_state)
        reward += self.tile_exploration_reward(game_state)
        reward += self.door_use_reward(
            action=action,
            game_state=game_state,
            scene_label=pre_scene_label,
            scene_confidence=pre_scene_confidence,
        )

        

        # -----------------------------------------------------
        # Keycard state placeholder
        # -----------------------------------------------------
        # Later, wire these to real shared-memory values if available.
        keys_owned = []

        for key_name in ["blue", "yellow", "red"]:
            if game_state.get(f"has_{key_name}_key", False):
                keys_owned.append(key_name)

        game_state["keys_owned"] = keys_owned

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

        reward += self.wall_sensor_reward(
            wall_sensor_state=wall_sensor_state,
            action=action,
            distance_moved=distance_moved,
            motion=motion,
        )

        game_state["distance_moved"] = distance_moved
        corner_trapped = self.detect_corner_trap(game_state)
        game_state["corner_trapped"] = corner_trapped

        director_result = self.route_director.evaluate(
            game_state=game_state,
            action=action,
            level_guide=self.level_guide,
        )

        reward += self.main_goal_progress_reward(director_result)

        reward += self.goal_progress_reward(director_result)

        reward += self.stagnation_penalty(
            game_state=game_state,
            director_result=director_result,
        )

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
        # Death review rolling buffer
        # -----------------------------------------------------
        if raw_frame is not None:
            self.recent_frame_buffer.append({
                "frame": raw_frame.copy(),
                "step": self._step_count,
                "action": action,
                "scene_label": scene_label,
                "scene_confidence": float(scene_confidence),
                "x": game_state.get("x"),
                "y": game_state.get("y"),
                "health": game_state.get("health"),
                "ammo": game_state.get("ammo"),
                "motion": float(motion),
                "distance_moved": float(distance_moved),
                "stuck_counter": int(self.stuck_counter),
                "wall_contact_steps": int(self.wall_contact_steps),
                "curriculum_stage": int(self.curriculum_stage),
            })

        # -----------------------------------------------------
        # Corridor milestone
        # -----------------------------------------------------
        # The first corridor is the first real danger checkpoint.
        # Reaching it consistently means the movement/route policy is good enough
        # to begin learning basic combat.
        # -----------------------------------------------------
        # Corridor milestone disabled
        # -----------------------------------------------------
        # Do not use hardcoded corridor coordinates as a curriculum trigger.
        # The agent should learn general navigation/survival instead.
        corridor_reached = False
        game_state["corridor_reached"] = False

        # -----------------------------------------------------
        # Corridor stay / corridor confidence reward
        # -----------------------------------------------------
        # Reaching the corridor is good, but the agent also needs
        # to learn not to immediately drift back into the spawn wall.
        # Corridor-specific reward disabled.
        # General movement, survival, object vision, wall avoidance, and goal progress
        # should train the behavior instead.

        # -----------------------------------------------------
        # Spawn wall / boundary return penalty
        # -----------------------------------------------------
        # This prevents the agent from repeatedly drifting back into
        # the known spawn-wall trap after it has already made progress.

        x = game_state.get("x")
        y = game_state.get("y")

        # -----------------------------------------------------
        # Hardcoded spawn-boundary logic disabled
        # -----------------------------------------------------
        # Use general wall/stuck/motion detection instead of map-specific coordinates.
        spawn_wall_zone = False
        game_state["spawn_wall_zone"] = False

        # -----------------------------------------------------
        # Secret / side-route memory milestone

        object_result = None

        if self.object_predictor is not None and raw_frame is not None:
            try:
                object_result = self.object_predictor.predict(raw_frame)

                print(
                    "[object_vision] "
                    f"present={object_result['present']} "
                    f"enemy={object_result['scores'].get('enemy_visible', 0.0):.2f} "
                    f"health={object_result['scores'].get('pickup_health', 0.0):.2f} "
                    f"ammo={object_result['scores'].get('pickup_ammo', 0.0):.2f} "
                    f"armor={object_result['scores'].get('pickup_armor', 0.0):.2f} "
                    f"barrel={object_result['scores'].get('explosive_barrel', 0.0):.2f} "
                    f"none={object_result['scores'].get('no_important_object', 0.0):.2f}"
                )

                game_state["object_vision"] = object_result

            except Exception as e:
                print(f"[object_vision] prediction failed: {e}")
                object_result = None
        # -----------------------------------------------------
        # Level-guide secret rewards
        # -----------------------------------------------------
        # Secrets are allowed, but not as one hardcoded first-level coordinate.
        # The active level guide provides all known secrets for the current map.
        secret_reached = False
        secret_name = None
        secret_reward = 0.0

        if not self.doom_has_focus():
            print("[doom_env] Doom lost focus. Truncating episode.")

            truncated = True
            info["doom_focus_lost"] = True
            self.last_shared_debug["doom_focus_lost"] = True

            self.controller.release_all()

            frame_cache.invalidate()
            safe_frame = self.observer.build()
            safe_observation = self.frame_stack.add_frame(safe_frame)

            self.episode_reward_total += float(reward)

            return (
                safe_observation,
                float(reward),
                terminated,
                truncated,
                info,
            )

        try:
            checkpoint_result = self.checkpoint_tracker.update(game_state)

            if isinstance(checkpoint_result, tuple):
                checkpoint_reward, checkpoint_info = checkpoint_result
            elif isinstance(checkpoint_result, dict):
                checkpoint_reward = 0.0
                checkpoint_info = checkpoint_result
            else:
                checkpoint_reward = 0.0
                checkpoint_info = {}

            reward += float(checkpoint_reward)

            secret_reached = checkpoint_info.get("secret_reached", False)
            secret_name = checkpoint_info.get("nearest_secret_name")
            secret_distance = checkpoint_info.get("distance_to_secret")

            checkpoint_reached = checkpoint_info.get("checkpoint_reached", False)
            checkpoint_name = checkpoint_info.get("checkpoint_name")
            checkpoint_distance = checkpoint_info.get("distance_to_checkpoint")

            game_state["secret_reached"] = secret_reached
            game_state["nearest_secret_name"] = secret_name
            game_state["distance_to_secret"] = secret_distance

            game_state["checkpoint_reached"] = checkpoint_reached
            game_state["checkpoint_name"] = checkpoint_name
            game_state["distance_to_checkpoint"] = checkpoint_distance

            if secret_reached:
                secret_key = secret_name or "unknown_secret"

                if not hasattr(self, "secrets_found_this_episode"):
                    self.secrets_found_this_episode = set()

                if secret_key not in self.secrets_found_this_episode:
                    self.secrets_found_this_episode.add(secret_key)

                    secret_reward = 6.0
                    reward += secret_reward
                    self.reward_manager.add("level_guide_secret_reached", secret_reward)

                    print(f"[secret] reached {secret_key}")

        except Exception as e:
            game_state["secret_reached"] = False
            game_state["nearest_secret_name"] = None
            game_state["distance_to_secret"] = None

            game_state["checkpoint_reached"] = False
            game_state["checkpoint_name"] = None
            game_state["distance_to_checkpoint"] = None

            checkpoint_reward = 0.0
            checkpoint_info = {}

            print(f"[tracker] checkpoint/secret tracker failed: {e}")

        # -----------------------------------------------------
        # Route loop escape shaping
        # -----------------------------------------------------
        tile_size = 64
        x = game_state.get("x")
        y = game_state.get("y")

        sensory_situation = game_state.get("sensory_situation")
        repeat = int(game_state.get("sensory_repeat", 0) or 0)

        if x is not None and y is not None:
            current_tile = (int(float(x) // tile_size), int(float(y) // tile_size))

            if sensory_situation == "stuck_or_looping":
                self.recent_loop_tile = current_tile
                self.recent_loop_steps = 20

                if repeat >= 10:
                    reward += self.add_penalty("slow_loop_escape", -0.02)
                    self.reward_manager.add("slow_loop_escape", -0.02)

            elif self.recent_loop_steps > 0:
                self.recent_loop_steps -= 1

                if self.recent_loop_tile is not None and current_tile != self.recent_loop_tile:
                    reward += 0.15
                    self.reward_manager.add("escaped_loop_tile", 0.15)
                    print(
                        f"[route_escape] escaped loop tile "
                        f"{self.recent_loop_tile} -> {current_tile}"
                    )

                    self.recent_loop_tile = None
                    self.recent_loop_steps = 0


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

        

        # -----------------------------------------------------
        # Strong wall / stuck punishment

        wall_info = self._wall_direction_info(raw_frame)

        reward += self.strong_wall_penalty(
            action=action,
            wall_info=wall_info,
            game_state=game_state,
        )

        game_state["left_wall_ratio"] = wall_info["left_ratio"]
        game_state["front_wall_ratio"] = wall_info["front_ratio"]
        game_state["right_wall_ratio"] = wall_info["right_ratio"]

        sensory_state = self.sensory_model.evaluate(
            game_state=game_state,
            scene_label=scene_label,
            scene_confidence=scene_confidence,
            wall_info=wall_info,
            action=action,
            distance_moved=distance_moved,
            motion=motion,
            stuck_counter=self.stuck_counter,
            wall_contact_steps=self.wall_contact_steps,
            corridor_reached=corridor_reached,
        )

        # -----------------------------------------------------
        # Sensory model logging/reward only
        # -----------------------------------------------------
        # Do not mutate action here. Doom already received the keypress.
        # Real sensory emergency control happens pre-action in sensory_emergency_action().
        sensory_recommended_action = sensory_state.get("recommended_action")
        game_state["sensory_recommended_action"] = sensory_recommended_action

        # -----------------------------------------------------
        # Retrace / opening search
        # -----------------------------------------------------
        # This is a general recovery planner:
        # back up, rotate toward open space, test forward, strafe search.
        # It prevents endless wall/object pushing without using hardcoded map positions.
        self.retrace_navigator.update_position(game_state)

        retrace_action = None

        if self.enable_retrace_navigator:
            retrace_action = self.retrace_navigator.get_action(
                sensory_state=sensory_state,
                wall_info=wall_info,
                distance_moved=distance_moved,
                motion=motion,
                stuck_counter=self.stuck_counter,
                wall_contact_steps=self.wall_contact_steps,
            )

        if retrace_action is not None and retrace_action in self.get_allowed_actions():
            before = action

            # Do not mutate action here for control purposes.
            # Doom already received the keypress earlier in this step.
            # Store it for the next pre-action step instead.
            self.pending_retrace_action = retrace_action
            self.retrace_lock_steps = max(self.retrace_lock_steps, 8)

            game_state["retrace_recommended_action"] = retrace_action

            print(
                f"[retrace] queued {before} -> {retrace_action} "
                f"phase={self.retrace_navigator.phase} "
                f"step={self.retrace_navigator.phase_step} "
                f"side={self.retrace_navigator.preferred_side} "
                f"lock={self.retrace_lock_steps}"
            )

        if self.retrace_navigator.active:
            reward += self.add_penalty("retrace_mode_active", -0.02)

        if (
            self.retrace_navigator.last_escape_action is not None
            and distance_moved > 6.0
            and motion > 2.0
        ):
            reward += 0.8
            self.reward_manager.add("successful_retrace_escape", 0.8)

        # -----------------------------------------------------
        # Sensory emergency logging only
        # -----------------------------------------------------
        # Do not mutate action here. Doom already received the keypress.
        # Real sensory emergency control happens pre-action.
        sensory_recommended_action = sensory_state.get("recommended_action")
        game_state["sensory_recommended_action"] = sensory_recommended_action

        director_state = self.route_director.evaluate(
            game_state=game_state,
            action=action,
            level_guide=self.level_guide,
        )

        game_state["director_target"] = director_state.get("target")
        game_state["director_distance"] = director_state.get("distance")
        game_state["director_dx"] = director_state.get("dx", 0.0)
        game_state["director_dy"] = director_state.get("dy", 0.0)
        game_state["director_hint_action"] = director_state.get("hint_action")
        game_state["director_objective"] = director_state.get("objective")
        game_state["director_reason"] = director_state.get("reason")

        reward += director_state.get("reward_delta", director_state.get("reward", 0.0))

        director_reward_delta = director_state.get("reward_delta", director_state.get("reward", 0.0))

        if director_reward_delta > 0:
            self.reward_manager.add("director_progress", director_reward_delta)
        elif director_reward_delta < 0:
            self.reward_manager.add("director_wrong_way", director_reward_delta)

        if self._step_count % 25 == 0:
            print(
                "[director] "
                f"target={director_state.get('target')} "
                f"reason={director_state.get('reason')} "
                f"dist={director_state.get('distance')} "
                f"delta={director_state.get('distance_delta', 0.0):.2f} "
                f"hint={director_state.get('hint_action')} "
                f"reward={director_state.get('reward_delta', director_state.get('reward', 0.0)):.2f}"
            )

        game_state["sensory_situation"] = sensory_state["situation"]
        game_state["sensory_confidence"] = sensory_state["confidence"]
        game_state["sensory_tile"] = sensory_state["tile"]
        game_state["sensory_repeated_tile_count"] = sensory_state["repeated_tile_count"]

        if sensory_state["reward_delta"] != 0.0:
            reward += sensory_state["reward_delta"]

            if sensory_state["reward_name"] is not None:
                self.reward_manager.add(
                    sensory_state["reward_name"],
                    sensory_state["reward_delta"],
                )

        if self._step_count % 25 == 0:
            print(
                "[sensory] "
                f"situation={sensory_state['situation']} "
                f"conf={sensory_state['confidence']:.2f} "
                f"tile={sensory_state['tile']} "
                f"repeat={sensory_state['repeated_tile_count']} "
                f"spawn={sensory_state['spawn_wall_zone']} "
                f"secret={sensory_state['secret_side_area']} "
                f"right_route={sensory_state['right_route_area']} "
                f"rec={sensory_state['recommended_action']}"
            )

        # -----------------------------------------------------
        # Strong wall / stuck punishment
        # -----------------------------------------------------
        # The agent has a habit of running into walls, wall-hugging,
        # backing into traps, and getting stuck near boundaries.
        # These penalties teach that walls are strongly bad unless the
        # agent is actively escaping them.

        pushing_forward_into_wall = (
            action == "move_forward"
            and (
                wall_contact
                or wall_info["front_wall"]
                or wall_info["front_ratio"] >= 0.45
                or (
                    scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall"]
                    and scene_confidence >= 0.70
                )
            )
        )

        low_movement_attempt = (
            action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]
            and distance_moved < 1.0
            and motion < 1.5
        )

        trapped_near_boundary = (
            scene_label == "boundary_or_stuck_wall"
            and scene_confidence >= 0.70
        )

        hugging_wall = (
            wall_info["front_ratio"] >= 0.55
            or wall_info["left_ratio"] >= 0.80
            or wall_info["right_ratio"] >= 0.80
        )

        if wall_contact:
            reward += self.add_penalty("wall_contact_strong", -3.0)

        if pushing_forward_into_wall:
            reward += self.add_penalty("push_into_wall_strong", -5.0)

        if low_movement_attempt:
            reward += self.add_penalty("low_movement_attempt", -1.5)

        if trapped_near_boundary:
            reward += self.add_penalty("boundary_trap_penalty", -2.5)

        if hugging_wall:
            reward += self.add_penalty("wall_hugging_strong", -1.0)

        if self.stuck_counter >= 5:
            reward += self.add_penalty("stuck_5_steps", -2.0)

        if self.stuck_counter >= 10:
            reward += self.add_penalty("stuck_10_steps", -4.0)

        if self.stuck_counter >= 20:
            reward += self.add_penalty("stuck_20_steps", -8.0)


        # Reward actual escape, but only if movement really improved.
        if self.stuck_counter >= 5 and distance_moved > 5.0 and motion > 2.0:
            reward += 3.0
            self.reward_manager.add("strong_escape_from_stuck", 3.0)

        # If the agent is near a wall but takes a reasonable escape action,
        # give a small reward so it learns the alternative.
        if trapped_near_boundary and action in ["turn_left", "turn_right", "strafe_left", "strafe_right", "move_backward"]:
            reward += 0.6
            self.reward_manager.add("correct_boundary_escape_action", 0.6)


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

        if self.corner_trap_steps >= 20 and not self.sensory_emergency_active:
            reward -= 5.0
            terminated = True
            info["corner_trap_reset"] = True
            print("[reset] corner trap timeout")

        # -----------------------------------------------------
        # Known use-point reward: doors, elevators, switches
        # -----------------------------------------------------
        near_use_point = False
        nearest_use_name = None

        try:
            px = game_state.get("x")
            py = game_state.get("y")

            if px is not None and py is not None:
                px = float(px)
                py = float(py)

                use_targets = (
                    self.level_guide.get("use_points", [])
                    + self.level_guide.get("switches", [])
                    + self.level_guide.get("locked_doors", [])
                )

                best_dist = None

                for use_target in use_targets:
                    tx = use_target.get("x")
                    ty = use_target.get("y")
                    radius = float(use_target.get("radius", 96.0))

                    if tx is None or ty is None:
                        continue

                    dx = float(tx) - px
                    dy = float(ty) - py
                    dist = (dx * dx + dy * dy) ** 0.5

                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        nearest_use_name = use_target.get("name")

                    if dist <= radius:
                        near_use_point = True
                        nearest_use_name = use_target.get("name")
                        break

        except Exception:
            near_use_point = False
            nearest_use_name = None

        game_state["near_use_point"] = near_use_point
        game_state["nearest_use_name"] = nearest_use_name
        reward += self.exit_distance_progress_reward(game_state)

        if near_use_point:
            if action == "use":
                reward += 2.0
                self.reward_manager.add("used_known_use_point", 2.0)
                print(f"[use_point] used {nearest_use_name}")

            elif action == "move_forward" and (
                scene_label in ["front_wall", "door_or_button", "obstacle"]
                or distance_moved < 1.0
            ):
                reward += self.add_penalty("ignored_use_point", -0.8)

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
                f"x={self._fmt_num(game_state.get('x'))} "
                f"y={self._fmt_num(game_state.get('y'))} "
                )

        if self._step_count % 100 == 0:
            print(
                f"[pos] x={self._fmt_num(game_state.get('x'))} "
                f"y={self._fmt_num(game_state.get('y'))} "
                f"shared={game_state.get('shared_state_available')}"
            )

            print(
                f"[nav] checkpoint={checkpoint_info.get('checkpoint_name')} "
                f"dist={checkpoint_info.get('distance_to_checkpoint')} "
                f"secret={checkpoint_info.get('nearest_secret_name')} "
                f"secret_dist={checkpoint_info.get('distance_to_secret')} "
                f"reward={checkpoint_reward:.2f}"
            )

        if self._step_count % 10 == 0:
            print(
                "[director_debug] "
                f"pos=({game_state.get('x'):.1f},{game_state.get('y'):.1f}) "
                f"target={director_result.get('target')} "
                f"target_xy=({director_result.get('target_x'):.1f},{director_result.get('target_y'):.1f}) "
                f"dx={director_result.get('dx'):.1f} "
                f"dy={director_result.get('dy'):.1f} "
                f"dist={director_result.get('distance'):.1f} "
                f"delta={director_result.get('distance_delta'):.1f} "
                f"hint={director_result.get('hint')} "
                f"hint_action={director_result.get('hint_action')}"
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

        if self.curriculum_stage < 3 and object_result is not None:
            object_result["present"] = [
                label for label in object_result.get("present", [])
                if label != "enemy_visible"
            ]
            object_result["scores"]["enemy_visible"] = 0.0

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
        # Ammo discipline reward shaping
        # -----------------------------------------------------
        ammo_discipline_reward = 0.0

        if action == "shoot":
            ammo = int(game_state.get("ammo", 0) or 0)

            if not enemy_visible:
                ammo_discipline_reward -= 0.03
                self.reward_manager.add("shoot_no_enemy", -0.03)

            elif enemy_visible and not enemy_centered:
                ammo_discipline_reward -= 0.02
                self.reward_manager.add("shoot_not_centered", -0.02)

            elif enemy_visible and enemy_centered:
                ammo_discipline_reward += 0.02
                self.reward_manager.add("shoot_centered_enemy", 0.02)

        reward += ammo_discipline_reward
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
                    else:
                        reward += 0.8
                        self.reward_manager.add("known_use_location_reused", 0.8)

                        print(f"[memory] useful use location discovered: {use_tile}")

            else:
                reward += self.add_penalty("use_spam_penalty", -1.0)

            if action == "swap_weapon":
                self.swap_weapon_count += 1

                weapon = self.normalize_weapon_name(current_weapon)
                object_scores = {}
                object_present = set()

                if object_result is not None:
                    object_scores = object_result.get("scores", {}) or {}
                    object_present = set(object_result.get("present", []) or [])

                barrel_visible = (
                    "explosive_barrel" in object_present
                    or float(object_scores.get("explosive_barrel", 0.0)) >= 0.35
                )

                object_enemy_score = float(object_scores.get("enemy_visible", 0.0))
                close_enemy = enemy_visible and object_enemy_score >= 0.70

                # Fists/melee are bad unless the enemy is very close.
                if weapon == "melee" and not close_enemy:
                    reward += self.add_penalty("bad_melee_weapon_distance", -1.0)

                if weapon == "melee" and close_enemy:
                    reward += 0.4
                    self.reward_manager.add("melee_only_close_range", 0.4)

                # RPG/rocket is powerful, but dangerous near barrels or close enemies.
                if weapon == "rpg" and (barrel_visible or close_enemy or health < 45):
                    reward += self.add_penalty("dangerous_rpg_context", -1.5)

                if weapon == "rpg" and enemy_visible and not barrel_visible and not close_enemy and health >= 60:
                    reward += 0.8
                    self.reward_manager.add("good_rpg_context", 0.8)

                # Shotgun/chaingun are generally good combat defaults.
                if enemy_visible and weapon in ["shotgun", "chaingun"]:
                    reward += 0.3
                    self.reward_manager.add("good_general_weapon", 0.3)

                # Weapon swap should not be spammed.
                if self.consecutive_swap_steps > 2:
                    reward += self.add_penalty("weapon_swap_spam", -0.8)

                x = game_state.get("x")
                y = game_state.get("y")

                if x is not None and y is not None:
                    current_tile = (int(float(x) // 64), int(float(y) // 64))

                    near_known_use = current_tile in self.secret_use_locations

                    if near_known_use and action == "move_forward" and scene_label in ["front_wall", "door_or_button"]:
                        reward += self.add_penalty("crashed_into_known_door", -2.0)

                    if near_known_use and action == "use":
                        reward += 1.0
                        self.reward_manager.add("used_known_door_location", 1.0)

        # -----------------------------------------------------
        # Weapon quality awareness
        # -----------------------------------------------------

        weapon_context_delta = self.weapon_context_reward(
            action=action,
            current_weapon=current_weapon,
            enemy_visible=enemy_visible,
            enemy_centered=enemy_centered,
            health=health,
            ammo=ammo,
            object_result=object_result,
        )

        reward += weapon_context_delta

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
        # Auxiliary-style object rewards
        # -----------------------------------------------------
        if object_result is not None:
            present_objects = set(object_result.get("present", []))
            scores = object_result.get("scores", {})

            if "enemy_visible" in present_objects:
                game_state["aux_enemy_visible"] = True
                reward += 0.02
                self.reward_manager.add("aux_enemy_awareness", 0.02)
            else:
                game_state["aux_enemy_visible"] = False

            if "pickup_health" in present_objects:
                game_state["aux_health_pickup_visible"] = True

                if health < 80 and action in ["move_forward", "strafe_left", "strafe_right"]:
                    reward += 0.08
                    self.reward_manager.add("aux_move_toward_health", 0.08)
            else:
                game_state["aux_health_pickup_visible"] = False

            if "pickup_ammo" in present_objects:
                game_state["aux_ammo_pickup_visible"] = True

                if ammo < 20 and action in ["move_forward", "strafe_left", "strafe_right"]:
                    reward += 0.08
                    self.reward_manager.add("aux_move_toward_ammo", 0.08)
            else:
                game_state["aux_ammo_pickup_visible"] = False

            if "explosive_barrel" in present_objects:
                game_state["aux_barrel_visible"] = True

                if action == "shoot" and enemy_visible:
                    reward += 0.05
                    self.reward_manager.add("aux_barrel_combat_awareness", 0.05)
            else:
                game_state["aux_barrel_visible"] = False


        # -----------------------------------------------------
        # Termination
        # -----------------------------------------------------

        if kill_delta > 0:
            print(
                f"[combat_success] kill gained "
                f"kill_delta={kill_delta} "
                f"total_kills={self.enemy_kill_count} "
                f"ammo={ammo} "
                f"health={health}"
            )

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
            self.episode_had_death = True
            self.save_death_review_frames(reason="death")

        if self.stuck_counter >= 35:
            reward -= 2.0
            terminated = True
            info["stuck_reset"] = True
            self.episode_had_stuck_reset = True
            self.save_death_review_frames(reason="stuck_reset")

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        reward += self.route_progress_reward(game_state)
        reward += self.goal_attraction_bubble_reward(game_state)
        #self.update_curriculum_from_progress(game_state)

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

        # Do not advance curriculum if the episode ended badly.
        # Reaching a corridor is not mastery if the agent dies there or gets stuck.
        if not info.get("death", False) and not info.get("stuck_reset", False):
            self.update_curriculum()
        else:
            print(
                "[Curriculum] Not advancing because episode ended with "
                f"death={info.get('death', False)} "
                f"stuck_reset={info.get('stuck_reset', False)}"
            )
        
        if self.use_shared_logic_reward:
            reward = float(self.last_shared_debug.get("shared_reward_preview", reward))

        self.episode_reward_total += float(reward)

        if terminated or truncated:
            if self.curriculum_stage >= 7:
                if self.episode_reward_total > self.best_episode_reward:
                    self.best_episode_reward = self.episode_reward_total
                    print(
                        f"[Mastery] New best episode reward: "
                        f"{self.best_episode_reward:.2f}"
                    )

        front_ratio = float(pre_wall_info.get("front_ratio", 0.0) or 0.0)
        left_ratio = float(pre_wall_info.get("left_ratio", 0.0) or 0.0)
        right_ratio = float(pre_wall_info.get("right_ratio", 0.0) or 0.0)

        front_safe = front_ratio < 0.28
        sides_safe = left_ratio < 0.38 and right_ratio < 0.38

        can_use_exploration_override = (
            self.retrace_lock_steps <= 0
            and not getattr(self.retrace_navigator, "active", False)
            and self.stuck_counter == 0
            and self.wall_contact_steps == 0
            and front_safe
            and sides_safe
        )

        if self.wall_contact_steps > 0 and action in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
            reward += 0.10
        self.reward_manager.add("wall_escape_attempt", 0.10)

        if front_ratio < 0.25 and left_ratio < 0.35 and right_ratio < 0.35:
            if self.wall_contact_steps > 0 or self.stuck_counter > 0:
                reward += 0.25
        self.reward_manager.add("cleared_wall_pressure", 0.25)

        info["route_progress_level"] = self.route_progress_level
        info["best_route_progress_level"] = self.best_route_progress_level
        info["unique_tiles"] = len(self.visited_tiles)
        info["doors_used"] = self.door_interaction_count
        info["stuck_counter"] = self.stuck_counter
        info["sensory_emergency_active"] = self.sensory_emergency_active

        shared_move_explore = self._shared_movement_reward_only()
        reward += shared_move_explore

        self.last_shared_debug["reward_after_shared_movement_exploration"] = float(reward)

        shared_wall = self._shared_wall_reward_only()
        reward += shared_wall

        self.last_shared_debug["reward_after_shared_wall"] = float(reward)

        shared_object_scene = self._shared_object_scene_reward_only()
        reward += shared_object_scene

        self.last_shared_debug["reward_after_shared_object_scene"] = float(reward)

        shared_route_curriculum = self._shared_route_curriculum_reward_only()
        reward += shared_route_curriculum

        self.last_shared_debug["reward_after_shared_route_curriculum"] = float(reward)

        # Small preview blend remains useful while full replacement is off.
        reward = self._blend_shared_reward(reward)

        self.last_shared_debug["reward_after_shared_blend_before_full_mode"] = float(reward)

        # Full shared reward mode is available but OFF by default.
        reward = self._full_shared_reward_replacement(reward)

        if death_like_screen:
            reward -= 25.0
            self.reward_manager.add("death_penalty", -25.0)
            terminated = True
            info["death"] = True
            self.episode_had_death = True

        if not self.training_mode:
            self.save_death_review_frames(reason="death")

        if self.stuck_counter >= 35:
            reward -= 2.0
            terminated = True
            info["stuck_reset"] = True
            self.episode_had_stuck_reset = True

        if not self.training_mode:
            self.save_death_review_frames(reason="stuck_reset")

        self.last_shared_debug["final_reward_output"] = float(reward)
        self.last_shared_debug["old_route_reward_scale"] = self.old_route_reward_scale
        self.last_shared_debug["old_curriculum_reward_scale"] = self.old_curriculum_reward_scale
        self.last_shared_debug["use_full_shared_reward_mode"] = self.use_full_shared_reward_mode

        return observation, float(reward), terminated, truncated, info

    # ---------------------------------------------------------
    # Action execution
    # ---------------------------------------------------------
    
    def save_death_review_frames(self, reason="death"):
        """
        Save the last few seconds before death/stuck reset.

        This does not train PPO immediately. It creates a dataset that can later
        be labeled as danger_scene, wall_stuck_under_fire, enemy_corridor_death,
        bad_open_path, etc.
        """
        if not self.recent_frame_buffer:
            return

        self.death_capture_count += 1

        death_dir = self.death_capture_dir / (
            f"{reason}_{self.death_capture_count:06d}_{int(time.time())}"
        )
        death_dir.mkdir(parents=True, exist_ok=True)

        metadata = []

        for idx, item in enumerate(self.recent_frame_buffer):
            frame = item.get("frame")

            if frame is not None:
                frame_path = death_dir / f"frame_{idx:03d}.jpg"

                try:
                    if frame.ndim == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(frame_path), frame_bgr)
                except Exception as e:
                    print(f"[death_review] failed saving frame {idx}: {e}")

            meta_item = {
                key: value
                for key, value in item.items()
                if key != "frame"
            }
            metadata.append(meta_item)

        with open(death_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[death_review] saved {len(metadata)} frames to {death_dir}")

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
            # Smart weapon select is handled in step() when full game_state/object_result
            # is available. If it reaches here, use the old fallback.
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

    def _build_shared_game_state(
            self,
            base_game_state,
            object_result=None,
            scene_result=None,
            hud_result=None,
        ):
            """
            Convert Doom Retro state + learned predictions into the same format
            used by ViZDoom and SharedDoomLogic.

            This does not control the game by itself.
            It only prepares normalized debug/reward data.
            """

            if base_game_state is None:
                base_game_state = {}

            shared_game_state = build_doomretro_game_state(
                base_game_state=base_game_state,
                object_prediction=object_result,
                scene_prediction=scene_result,
                hud_prediction=hud_result,
            )

            return shared_game_state
        
    def exit_distance_progress_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        main_goal = self.level_guide.get("main_goal")
        if not main_goal:
            return 0.0

        x = float(x)
        y = float(y)
        gx = float(main_goal["x"])
        gy = float(main_goal["y"])

        dist = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

        if not hasattr(self, "last_exit_distance"):
            self.last_exit_distance = None

        if not hasattr(self, "best_exit_distance"):
            self.best_exit_distance = None

        reward = 0.0

        if self.last_exit_distance is not None:
            improvement = self.last_exit_distance - dist

            if improvement > 1.0:
                reward += min(0.08, improvement / 128.0)
                self.reward_manager.add("exit_distance_closer", reward)

            elif improvement < -4.0:
                penalty = min(0.10, abs(improvement) / 128.0)
                reward -= penalty
                self.reward_manager.add("exit_distance_farther", -penalty)

        if self.best_exit_distance is None or dist < self.best_exit_distance - 8.0:
            self.best_exit_distance = dist
            reward += 0.10
            self.reward_manager.add("new_best_exit_distance", 0.10)

        self.last_exit_distance = dist

        return reward

   
    def normalize_weapon_name(self, weapon_name):
        weapon = str(weapon_name or "").lower()

        if "fist" in weapon or "chainsaw" in weapon or "ripper" in weapon or "ripter" in weapon:
            return "melee"

        if "pistol" in weapon:
            return "pistol"

        if "shotgun" in weapon or "scatter" in weapon:
            return "shotgun"

        if "chaingun" in weapon or "machine" in weapon or "rifle" in weapon:
            return "chaingun"

        if "rocket" in weapon or "rpg" in weapon or "launcher" in weapon:
            return "rpg"

        if "plasma" in weapon:
            return "plasma"

        if "bfg" in weapon:
            return "bfg"

        return weapon or "unknown"
    
    def _fmt_num(self, value, digits=1):
        try:
            if value is None:
                return "None"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "None"
        
    def fast_enemy_reaction_action(self, action, game_state):
        """
        Fast emergency combat reaction.

        Purpose:
        - shoot immediately when enemy is centered and ammo exists
        - turn toward enemy quickly when off-center
        - dodge/retreat instead of fighting with 0 ammo
        """

        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        enemy_left = bool(game_state.get("enemy_left", False))
        enemy_right = bool(game_state.get("enemy_right", False))
        ammo = int(game_state.get("ammo", 0) or 0)
        health = int(game_state.get("health", 100) or 100)

        if not enemy_visible:
            return action

        # Do not let combat interrupt retrace/wall recovery.
        if (
            self.retrace_lock_steps > 0
            or getattr(self.retrace_navigator, "active", False)
            or self.stuck_counter >= 4
        ):
            return action

        # No ammo = do not fight. Move.
        if ammo <= 0:
            if health <= 35:
                return "move_backward"

            if self._step_count % 2 == 0:
                return "strafe_left"
            return "strafe_right"

        # Enemy centered + ammo = shoot immediately.
        if enemy_centered:
            return "shoot"

        # Enemy not centered = snap aim faster.
        if enemy_left:
            return "turn_left"

        if enemy_right:
            return "turn_right"

        return action
        
    def _shared_object_scene_reward_only(self):
        """
        Extract only object/scene shaping from SharedDoomLogic debug.

        This uses Doom Retro ObjectPredictor / ScenePredictor indirectly
        through perception.prediction_adapter.build_doomretro_game_state().
        """

        if not self.use_shared_object_scene_reward:
            return 0.0

        debug = self.last_shared_debug or {}

        object_scene = debug.get("object_scene_reward", 0.0)
        combat = debug.get("combat_reward", 0.0)
        use_reward = debug.get("use_reward", 0.0)

        try:
            object_scene = float(object_scene)
        except Exception:
            object_scene = 0.0

        try:
            combat = float(combat)
        except Exception:
            combat = 0.0

        try:
            use_reward = float(use_reward)
        except Exception:
            use_reward = 0.0

        total = (object_scene + combat + use_reward) * self.shared_object_scene_reward_scale

        debug["shared_object_scene_component"] = object_scene
        debug["shared_combat_component"] = combat
        debug["shared_use_component"] = use_reward
        debug["shared_object_scene_total"] = total

        self.last_shared_debug = debug

        return float(total)
    
    def goal_heading_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")
        angle = game_state.get("angle")

        if x is None or y is None or angle is None:
            return 0.0

        target = self.get_active_goal_bubble_target(game_state)

        if target is None:
            return 0.0

        x = float(x)
        y = float(y)
        angle = float(angle)

        tx = float(target["x"])
        ty = float(target["y"])

        target_angle = math.degrees(math.atan2(ty - y, tx - x))
        diff = abs((target_angle - angle + 180.0) % 360.0 - 180.0)

        reward = 0.0

        if diff <= 15:
            reward += 0.05
            self.reward_manager.add("facing_goal", 0.05)
        elif diff >= 90:
            reward -= 0.05
            self.reward_manager.add("facing_away_from_goal", -0.05)

        return reward
    
    def main_goal_progress_reward(self, director_result):
        """
        Reward getting closer to the real level exit.

        This should be stronger and more consistent than checkpoint rewards.
        """

        if director_result is None:
            return 0.0

        distance = director_result.get("distance")
        distance_delta = float(director_result.get("distance_delta", 0.0) or 0.0)
        hint = director_result.get("hint")

        if distance is None:
            return 0.0

        reward = 0.0

        # Moving closer to the real exit.
        if distance_delta > 1.0:
            progress_reward = min(0.12, distance_delta / 96.0)
            reward += progress_reward
            self.reward_manager.add("main_goal_closer", progress_reward)

        # New best distance to exit.
        if not hasattr(self, "best_main_goal_distance"):
            self.best_main_goal_distance = None

        if self.best_main_goal_distance is None:
            self.best_main_goal_distance = float(distance)

        if float(distance) < self.best_main_goal_distance - 8.0:
            self.best_main_goal_distance = float(distance)
            reward += 0.10
            self.reward_manager.add("main_goal_new_best", 0.10)

        # Moving away from the exit.
        if distance_delta < -6.0:
            penalty = min(0.12, abs(distance_delta) / 96.0)
            reward -= penalty
            self.reward_manager.add("main_goal_moving_away", -penalty)

        # Stuck while not progressing.
        if hint == "recover_unstuck":
            reward -= 0.08
            self.reward_manager.add("main_goal_stuck", -0.08)

        # Reached the actual exit area.
        if hint == "goal_reached":
            reward += 10.0
            self.reward_manager.add("main_goal_reached", 10.0)

        return reward
        
    def _shared_wall_reward_only(self):
        """
        Extract only wall/stuck/depth-style reward from SharedDoomLogic debug.

        For Doom Retro, depth_obs is usually None, but SharedDoomLogic can still
        use scene/object/wall-style signals from the normalized game_state.
        """

        if not self.use_shared_wall_reward:
            return 0.0

        debug = self.last_shared_debug or {}

        wall = debug.get("wall_reward", 0.0)

        try:
            wall = float(wall)
        except Exception:
            wall = 0.0

        total = wall * self.shared_wall_reward_scale

        debug["shared_wall_component"] = wall
        debug["shared_wall_total"] = total

        self.last_shared_debug = debug

        return float(total)
    
    def combat_movement_reward(self, game_state, action, distance_moved):
        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        ammo = int(game_state.get("ammo", 0) or 0)
        health = int(game_state.get("health", 100) or 100)

        reward = 0.0
        moved = float(distance_moved or 0.0)

        if not enemy_visible:
            return 0.0

        # Standing still in combat is bad.
        if moved < 2.0 and action in ["turn_left", "turn_right", "shoot", "melee_attack"]:
            reward -= 0.12
            self.reward_manager.add("combat_standing_still", -0.12)

        # Strafing/backing up while enemy is visible is useful.
        if action in ["strafe_left", "strafe_right"] and moved >= 2.0:
            reward += 0.08
            self.reward_manager.add("combat_strafe_movement", 0.08)

        if action == "move_backward" and enemy_visible:
            reward += 0.05
            self.reward_manager.add("combat_retreat_spacing", 0.05)

        # Shooting rules.
        if action == "shoot" and ammo <= 0:
            reward -= 0.60
            self.reward_manager.add("shoot_no_ammo", -0.60)

        elif action == "shoot" and enemy_centered and ammo > 0:
            reward += 0.30
            self.reward_manager.add("shoot_centered_with_ammo", 0.30)

        elif action == "shoot" and enemy_visible and not enemy_centered and ammo > 0:
            reward -= 0.15
            self.reward_manager.add("shoot_not_centered", -0.15)

        # Low health means movement is more important.
        if health <= 35 and action in ["strafe_left", "strafe_right", "move_backward"]:
            reward += 0.10
            self.reward_manager.add("low_health_dodge", 0.10)

        return reward
    
    def weapon_ammo_policy_reward(self, game_state, action):
        weapon = str(
            game_state.get("weapon")
            or game_state.get("selected_weapon")
            or self.current_weapon_name
            or "pistol"
        ).lower()

        ammo = int(game_state.get("ammo", 0) or 0)
        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))

        reward = 0.0

        rare_ammo_weapons = [
            "rocket",
            "rocket_launcher",
            "plasma",
            "plasma_rifle",
            "bfg",
        ]

        common_ammo_weapons = [
            "pistol",
            "chaingun",
            "shotgun",
            "super_shotgun",
        ]

        using_rare_ammo = any(name in weapon for name in rare_ammo_weapons)
        using_common_ammo = any(name in weapon for name in common_ammo_weapons)

        if action == "shoot" and ammo <= 0:
            reward -= 0.75
            self.reward_manager.add("weapon_no_ammo_shot", -0.75)
            return reward

        if action == "shoot" and using_rare_ammo:
            if enemy_visible and enemy_centered:
                reward += 0.10
                self.reward_manager.add("rare_ammo_good_shot", 0.10)
            else:
                reward -= 0.35
                self.reward_manager.add("rare_ammo_wasted", -0.35)

        elif action == "shoot" and using_common_ammo:
            if enemy_visible and enemy_centered:
                reward += 0.15
                self.reward_manager.add("common_ammo_good_shot", 0.15)
            elif not enemy_visible:
                reward -= 0.12
                self.reward_manager.add("common_ammo_wasted", -0.12)

        return reward
        
    def _shared_movement_reward_only(self):
        """
        Extract only movement/exploration reward from SharedDoomLogic debug.

        This lets Doom Retro gradually migrate reward components without
        replacing the entire reward function at once.
        """

        if not self.use_shared_movement_reward:
            return 0.0

        debug = self.last_shared_debug or {}

        movement = debug.get("movement_reward", 0.0)
        exploration = debug.get("exploration_reward", 0.0)

        try:
            movement = float(movement)
        except Exception:
            movement = 0.0

        try:
            exploration = float(exploration)
        except Exception:
            exploration = 0.0

        total = (movement + exploration) * self.shared_movement_reward_scale

        debug["shared_movement_component"] = movement
        debug["shared_exploration_component"] = exploration
        debug["shared_movement_exploration_total"] = total

        self.last_shared_debug = debug

        return float(total)
    
    def _shared_route_curriculum_reward_only(self):
        """
        Extract route/curriculum shaping from SharedDoomLogic debug.

        route_reward comes from SharedDoomLogic route zones.
        curriculum_stage is debug/status, not usually a direct reward by itself.
        """

        if not self.use_shared_route_curriculum_reward:
            return 0.0

        debug = self.last_shared_debug or {}

        route = debug.get("route_reward", 0.0)

        try:
            route = float(route)
        except Exception:
            route = 0.0

        total = route * self.shared_route_curriculum_reward_scale

        debug["shared_route_component"] = route
        debug["shared_route_curriculum_total"] = total
        debug["shared_curriculum_stage"] = debug.get("curriculum_stage")
        debug["shared_curriculum_weights"] = debug.get("curriculum_weights")

        self.last_shared_debug = debug

        return float(total)
    
    def _full_shared_reward_replacement(self, base_reward):
        """
        Optional full replacement mode.

        This replaces Doom Retro's reward with SharedDoomLogic's reward preview.
        It is OFF by default because it is the riskiest migration step.
        """

        if not self.use_full_shared_reward_mode:
            return float(base_reward)

        debug = self.last_shared_debug or {}
        shared_reward = debug.get("shared_reward_preview")

        try:
            shared_reward = float(shared_reward)
        except Exception:
            return float(base_reward)

        scaled = shared_reward * self.full_shared_reward_scale

        clipped = max(
            -self.full_shared_reward_clip,
            min(self.full_shared_reward_clip, scaled),
        )

        debug["full_shared_reward_raw"] = shared_reward
        debug["full_shared_reward_scaled"] = scaled
        debug["full_shared_reward_clipped"] = clipped
        debug["base_reward_replaced_by_full_shared"] = float(base_reward)

        self.last_shared_debug = debug

        return float(clipped)

    def choose_weapon_key(self, game_state, object_result=None, enemy_visible=False, enemy_centered=False):
        """
        Smart weapon selector.

        The action is still 'swap_weapon', but it no longer blindly cycles.
        It chooses a weapon key based on range, health, ammo, and barrel danger.

        Doom/Freedoom-style defaults:
        1 = fist/chainsaw/ripper
        2 = pistol
        3 = shotgun
        4 = chaingun
        5 = rocket/rpg
        6 = plasma
        7 = bfg
        """

        current_weapon = self.normalize_weapon_name(game_state.get("weapon", ""))
        ammo = int(game_state.get("ammo", 0) or 0)
        health = int(game_state.get("health", 100) or 100)

        scores = {}
        present = set()

        if object_result is not None:
            scores = object_result.get("scores", {}) or {}
            present = set(object_result.get("present", []) or [])

        barrel_visible = (
            "explosive_barrel" in present
            or float(scores.get("explosive_barrel", 0.0)) >= 0.35
        )

        object_enemy_score = float(scores.get("enemy_visible", 0.0))
        object_enemy_visible = (
            "enemy_visible" in present
            or object_enemy_score >= 0.55
        )

        enemy_seen = bool(enemy_visible or object_enemy_visible)

        # Your object model score is a rough proxy for enemy closeness.
        # Higher score usually means the enemy occupies more of the frame.
        enemy_close = enemy_seen and object_enemy_score >= 0.70
        enemy_mid = enemy_seen and object_enemy_score >= 0.45

        # No visible enemy: use a safe general-purpose weapon, not fists.
        if not enemy_seen:
            if current_weapon in ["melee", "unknown"]:
                return "2", "safe_default_pistol"
            return None, "keep_current_no_enemy"

        # Close quarters: ripper/chainsaw/fist can be okay, but only close.
        if enemy_close and health >= 60:
            # If you have the ripper/chainsaw equipped, keep it.
            if current_weapon == "melee":
                return None, "keep_melee_close_range"

            # Shotgun is safer than RPG in close quarters.
            return "3", "close_enemy_shotgun"

        # Never use RPG/rocket near barrels or in close range.
        if barrel_visible or enemy_close or health < 45:
            # RPG/rocket is dangerous near barrels, close enemies, or low health.
            if current_weapon == "rpg" and (barrel_visible or enemy_close or health < 45):
                return "3", "avoid_rpg_self_damage"
            return "3", "safe_shotgun"
   
        # Only keep RPG for safer long-range situations.
        if current_weapon == "rpg":
            if enemy_seen and not enemy_close and not barrel_visible and health >= 60:
                return None, "keep_rpg_safe_range"
            return "3", "rpg_not_safe_switch_shotgun"

        # Mid-range enemy: shotgun or chaingun.
        if enemy_mid:
            if current_weapon in ["shotgun", "chaingun"]:
                return None, "keep_midrange_weapon"

            return "3", "midrange_shotgun"

        # Far enemy / open area: chaingun or pistol.
        if ammo > 10:
            if current_weapon in ["chaingun", "shotgun"]:
                return None, "keep_good_weapon"

            return "4", "far_enemy_chaingun"

        # Low ammo fallback.
        if current_weapon == "melee":
            return "2", "low_ammo_pistol"

        return None, "keep_current"
    
    def action_prior_advice_reward(self, frame, action):
        """
        Advice-only reward from the ViZDoom teacher-trained action prior.

        This does NOT override PPO. It only gives a tiny reward when PPO's
        action agrees with the teacher-student action prior.
        """

        if not self.use_action_prior_advice:
            return 0.0, None

        if self.action_prior_advisor is None:
            return 0.0, None

        result = self.action_prior_advisor.predict(frame)

        if not result.get("enabled", False):
            return 0.0, result

        advised_action = result.get("action_name")
        confidence = float(result.get("confidence", 0.0))

        reward = 0.0

        if confidence >= self.action_prior_min_confidence:
            if advised_action == action:
                reward += self.action_prior_reward_scale
            else:
                reward -= self.action_prior_reward_scale * 0.25

        return float(reward), result

    def route_progress_reward(self, game_state):
        """
        Real-map route progress.

        This uses route_zones from navigation/level_guides.py.
        No hardcoded old route names are allowed here.
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        route_zones = self.level_guide.get("route_zones", [])

        if not route_zones:
            return 0.0

        reward = 0.0

        for idx, zone in enumerate(route_zones):
            name = zone.get("name", f"route_zone_{idx}")
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            radius = float(zone.get("radius", 128.0))
            zone_reward = float(zone.get("reward", 0.05))

            if name in self.route_zones_reached:
                continue

            dx = x - zx
            dy = y - zy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= radius:
                self.route_zones_reached.add(name)
                self.route_progress_level = max(self.route_progress_level, idx + 1)
                reward += zone_reward

                print(
                    f"[route_progress] reached={name} "
                    f"level={self.route_progress_level} "
                    f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
                )

        return float(reward)


    def doom_has_focus(self):
        try:
            expected = str(getattr(self.controller, "window_id", "")).strip()

            if not expected:
                return False

            active = subprocess.check_output(
                ["xdotool", "getactivewindow"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            return active == expected

        except Exception:
            return True

    def smart_weapon_select(self, game_state, object_result=None, enemy_visible=False, enemy_centered=False):
        """
        Execute smart weapon selection with a cooldown.

        This prevents endless weapon cycling and avoids getting stuck on fists.
        """
        if self._step_count - self.last_weapon_select_step < self.weapon_select_cooldown:
            return False

        key, reason = self.choose_weapon_key(
            game_state=game_state,
            object_result=object_result,
            enemy_visible=enemy_visible,
            enemy_centered=enemy_centered,
        )

        if key is None:
            if self._step_count % 25 == 0:
                print(f"[weapon_select] keep current reason={reason}")
            return False

        self.controller.release_all()
        self.controller.hold_key(key, duration=0.08)

        self.last_weapon_select_step = self._step_count
        self.last_weapon_key = key

        print(f"[weapon_select] key={key} reason={reason}")

        return True

    def _visual_area_signature(self, frame):
        if frame is None or frame.size == 0:
            return None

        small = frame[::8, ::8, :].astype(np.int16)
        quantized = small // 32

        return hash(quantized.tobytes())
    
    def _blend_shared_reward(self, base_reward):
        """
        Blend a small amount of SharedDoomLogic reward into the existing
        Doom Retro reward.

        This is safer than fully replacing the reward.
        """

        if not self.use_shared_logic_reward_blend:
            return float(base_reward)

        shared_reward = self.last_shared_debug.get("shared_reward_preview")

        try:
            shared_reward = float(shared_reward)
        except Exception:
            return float(base_reward)

        scaled = shared_reward * self.shared_logic_reward_scale

        clipped = max(
            -self.shared_logic_reward_clip,
            min(self.shared_logic_reward_clip, scaled),
        )

        self.last_shared_debug["shared_reward_scaled"] = scaled
        self.last_shared_debug["shared_reward_clipped"] = clipped
        self.last_shared_debug["base_reward_before_shared_blend"] = float(base_reward)
        self.last_shared_debug["final_reward_after_shared_blend"] = float(base_reward + clipped)

        return float(base_reward + clipped)
    
    def strong_wall_penalty(self, action, wall_info, game_state):
        """
        Strong temporary wall penalty.

        Purpose:
        - punish moving into walls
        - punish staying glued to walls
        - punish repeated wall contact
        - still allow turning/backing up as recovery
        """

        if not wall_info:
            return 0.0

        reward = 0.0

        left_ratio = float(wall_info.get("left_ratio", 0.0) or 0.0)
        front_ratio = float(wall_info.get("front_ratio", 0.0) or 0.0)
        right_ratio = float(wall_info.get("right_ratio", 0.0) or 0.0)

        front_wall = bool(wall_info.get("front_wall", False)) or front_ratio >= 0.35
        side_wall = left_ratio >= 0.35 or right_ratio >= 0.35
        heavy_wall_pressure = front_ratio >= 0.45 or left_ratio >= 0.50 or right_ratio >= 0.50

        sensory_situation = game_state.get("sensory_situation")
        stuck_or_looping = sensory_situation == "stuck_or_looping"

        # 1. Biggest mistake: moving forward into a front wall.
        if action == "move_forward" and front_wall:
            reward -= 1.00
            self.reward_manager.add("strong_forward_into_wall", -1.00)

        # 2. Strong wall pressure near any side.
        if heavy_wall_pressure:
            reward -= 0.50
            self.reward_manager.add("strong_wall_pressure", -0.50)

        # 3. Side scraping is bad, but less bad than front collision.
        elif side_wall:
            reward -= 0.25
            self.reward_manager.add("side_wall_scrape", -0.25)

        # 4. Repeated wall contact gets worse over time.
        if self.wall_contact_steps >= 2:
            penalty = min(1.50, 0.20 * self.wall_contact_steps)
            reward -= penalty
            self.reward_manager.add("repeated_wall_contact", -penalty)

        # 5. Wall + stuck loop is the worst case.
        if stuck_or_looping and (front_wall or side_wall):
            reward -= 1.00
            self.reward_manager.add("stuck_against_wall", -1.00)

        # 6. Do not punish recovery actions as hard.
        if action in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
            reward += 0.10
            self.reward_manager.add("wall_recovery_action", 0.10)

        return reward
    
    def goal_attraction_bubble_reward(self, game_state):
        """
        Reverse wall-bubble reward.

        The agent gets:
        - small reward for being in the outer target bubble
        - bigger reward for entering closer rings
        - progress reward for reducing distance
        - penalty for moving away after entering the bubble
        - penalty for spinning/stalling inside the bubble
        """

        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        target = self.get_active_goal_bubble_target(game_state)

        if target is None:
            return 0.0

        tx = float(target["x"])
        ty = float(target["y"])
        target_name = target.get("name", "unknown_goal")

        dx = x - tx
        dy = y - ty
        dist = (dx * dx + dy * dy) ** 0.5

        reward = 0.0

        # Track best distance to this target.
        if not hasattr(self, "goal_bubble_best_dist"):
            self.goal_bubble_best_dist = {}

        if not hasattr(self, "goal_bubble_last_dist"):
            self.goal_bubble_last_dist = {}

        if not hasattr(self, "goal_bubble_stall_steps"):
            self.goal_bubble_stall_steps = {}

        best_dist = self.goal_bubble_best_dist.get(target_name)
        last_dist = self.goal_bubble_last_dist.get(target_name)

        if best_dist is None:
            best_dist = dist
            self.goal_bubble_best_dist[target_name] = dist

        if last_dist is None:
            last_dist = dist

        improvement = last_dist - dist

        # -----------------------------
        # Ring reward: closer = bigger
        # -----------------------------
        if dist <= 1024:
            reward += 0.01
            self.reward_manager.add("goal_outer_bubble", 0.01)

        if dist <= 768:
            reward += 0.02
            self.reward_manager.add("goal_mid_outer_bubble", 0.02)

        if dist <= 512:
            reward += 0.04
            self.reward_manager.add("goal_mid_bubble", 0.04)

        if dist <= 256:
            reward += 0.08
            self.reward_manager.add("goal_inner_bubble", 0.08)

        if dist <= 128:
            reward += 0.15
            self.reward_manager.add("goal_core_bubble", 0.15)

        # -----------------------------
        # Delta reward: moving closer
        # -----------------------------
        if improvement > 2.0:
            progress_reward = min(0.12, improvement / 128.0)
            reward += progress_reward
            self.reward_manager.add("goal_bubble_closer", progress_reward)

            self.goal_bubble_stall_steps[target_name] = 0

        elif improvement < -4.0:
            penalty = min(0.20, abs(improvement) / 96.0)
            reward -= penalty
            self.reward_manager.add("goal_bubble_leaving", -penalty)

        else:
            self.goal_bubble_stall_steps[target_name] = (
                self.goal_bubble_stall_steps.get(target_name, 0) + 1
            )

        # -----------------------------
        # Best-distance reward
        # -----------------------------
        if dist < best_dist - 8.0:
            self.goal_bubble_best_dist[target_name] = dist
            reward += 0.10
            self.reward_manager.add("goal_bubble_new_best", 0.10)

        # -----------------------------
        # Anti-spin inside bubble
        # -----------------------------
        stall_steps = self.goal_bubble_stall_steps.get(target_name, 0)

        if dist <= 512 and stall_steps >= 8:
            penalty = min(0.30, 0.03 * stall_steps)
            reward -= penalty
            self.reward_manager.add("goal_bubble_spin_stall", -penalty)

        self.goal_bubble_last_dist[target_name] = dist

        return reward
    
    def get_active_goal_bubble_target(self, game_state):
        """
        Pick the next unreached route target.

        This prevents the agent from camping the corridor.
        Once a zone is reached, the active bubble moves forward.
        """

        route_zones = self.level_guide.get("route_zones", [])

        reached = getattr(self, "route_zones_reached", set())

        for zone in route_zones:
            name = zone.get("name")

            if name not in reached:
                return zone

        main_goal = self.level_guide.get("main_goal")

        if main_goal is not None:
            return main_goal

        return None

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
    
    def stagnation_penalty(self, game_state, director_result=None):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        reward = 0.0

        if self.last_progress_position is not None:
            old_x, old_y = self.last_progress_position
            moved = ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5

            if moved < 2.0:
                self.no_position_change_steps += 1
            else:
                self.no_position_change_steps = 0

            if self.no_position_change_steps >= 4:
                penalty = min(1.0, 0.10 * self.no_position_change_steps)
                reward -= penalty
                self.reward_manager.add("no_position_change", -penalty)

                if self.no_position_change_steps % 5 == 0:
                    print(
                        f"[stagnation] no_position_change "
                        f"steps={self.no_position_change_steps} "
                        f"moved={moved:.2f} "
                        f"penalty={penalty:.2f}"
                    )

        self.last_progress_position = (x, y)

        if director_result is not None:
            distance = director_result.get("distance")

            if distance is not None:
                distance = float(distance)

                if self.last_distance_to_goal is not None:
                    improvement = self.last_distance_to_goal - distance

                    if improvement < 1.0:
                        self.no_distance_progress_steps += 1
                    else:
                        self.no_distance_progress_steps = 0

                    if self.no_distance_progress_steps >= 8:
                        penalty = min(0.75, 0.05 * self.no_distance_progress_steps)
                        reward -= penalty
                        self.reward_manager.add("no_goal_distance_progress", -penalty)

                        if self.no_distance_progress_steps % 5 == 0:
                            print(
                                f"[stagnation] no_goal_distance_progress "
                                f"steps={self.no_distance_progress_steps} "
                                f"improvement={improvement:.2f} "
                                f"penalty={penalty:.2f}"
                            )

                self.last_distance_to_goal = distance

        return reward

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
        if self.curriculum_stage >= self.max_stage:
            return

        survived_enough = self._step_count >= 500
        moved_enough = self.distance_traveled >= 300.0
        explored_enough = len(self.visited_tiles) >= 8
        not_too_stuck = self.stuck_counter < 15
        no_bad_end = (
            not getattr(self, "episode_had_death", False)
            and not getattr(self, "episode_had_stuck_reset", False)
        )

        if self.curriculum_stage == 1:
            stage_passed = (
                survived_enough
                and moved_enough
                and explored_enough
                and not_too_stuck
                and no_bad_end
            )

        elif self.curriculum_stage == 2:
            stage_passed = (
                survived_enough
                and moved_enough
                and explored_enough
                and self.door_interaction_count >= 1
                and not_too_stuck
                and no_bad_end
            )

        elif self.curriculum_stage == 3:
            stage_passed = (
                survived_enough
                and moved_enough
                and self.valid_shot_count >= 3
                and self.combat_survival_steps >= 300
                and not_too_stuck
                and no_bad_end
            )

        else:
            stage_passed = False

        if stage_passed:
            old_stage = self.curriculum_stage
            self.curriculum_stage += 1

            print(
                f"[Curriculum] ADVANCING FROM STAGE "
                f"{old_stage} → {self.curriculum_stage}"
            )

    # ---------------------------------------------------------
    # Combat / action helpers
    # ---------------------------------------------------------
    def combat_readiness_reward(self, game_state, action):
        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        ammo = int(game_state.get("ammo", 0) or 0)

        reward = 0.0

        if enemy_visible and ammo > 0:
            reward += 0.03
            self.reward_manager.add("enemy_seen_with_ammo", 0.03)

        if enemy_visible and enemy_centered and ammo > 0:
            reward += 0.08
            self.reward_manager.add("enemy_centered_with_ammo", 0.08)

        if action == "shoot" and enemy_visible and enemy_centered and ammo > 0:
            reward += 0.20
            self.reward_manager.add("valid_shoot_attempt", 0.20)

        if action == "shoot" and not enemy_visible:
            reward -= 0.10
            self.reward_manager.add("bad_shoot_no_enemy", -0.10)

        return reward

    def weapon_context_reward(
        self,
        action,
        current_weapon,
        enemy_visible,
        enemy_centered,
        health,
        ammo,
        object_result=None,
    ):
        """
        Reward the agent for using the right weapon context.

        This is adaptive:
        - It does not assume one specific level.
        - It rewards categories: melee, pistol, shotgun, chaingun, rpg, plasma, bfg.
        - It discourages blind weapon swapping.
        """

        reward = 0.0
        weapon = self.normalize_weapon_name(current_weapon)

        scores = {}
        present = set()

        if object_result is not None:
            scores = object_result.get("scores", {}) or {}
            present = set(object_result.get("present", []) or [])

        object_enemy_score = float(scores.get("enemy_visible", 0.0))
        barrel_score = float(scores.get("explosive_barrel", 0.0))

        object_enemy_visible = (
            "enemy_visible" in present
            or object_enemy_score >= 0.55
        )

        barrel_visible = (
            "explosive_barrel" in present
            or barrel_score >= 0.35
        )

        trusted_enemy_visible = bool(enemy_visible or object_enemy_visible)

        enemy_close = trusted_enemy_visible and object_enemy_score >= 0.70
        enemy_mid_or_far = trusted_enemy_visible and not enemy_close

        # -----------------------------------------------------
        # Good weapon contexts
        # -----------------------------------------------------

        if trusted_enemy_visible and weapon in ["pistol", "shotgun", "chaingun", "plasma"]:
            reward += 0.20
            self.reward_manager.add("good_combat_weapon_context", 0.20)

        if enemy_close and weapon == "melee":
            reward += 0.30
            self.reward_manager.add("melee_close_context", 0.30)

        if enemy_mid_or_far and weapon in ["shotgun", "chaingun", "plasma"]:
            reward += 0.25
            self.reward_manager.add("midrange_weapon_context", 0.25)

        if weapon == "rpg":
            if trusted_enemy_visible and not enemy_close and not barrel_visible and health >= 60:
                reward += 0.30
                self.reward_manager.add("safe_rpg_context", 0.30)
            else:
                reward += self.add_penalty("unsafe_rpg_context", -0.80)

        # -----------------------------------------------------
        # Bad weapon contexts
        # -----------------------------------------------------

        if weapon == "melee" and enemy_mid_or_far:
            reward += self.add_penalty("bad_melee_distance", -0.70)

        if weapon == "melee" and not trusted_enemy_visible:
            reward += self.add_penalty("unneeded_melee_weapon", -0.25)

        if weapon in ["unknown", ""]:
            reward += self.add_penalty("unknown_weapon_state", -0.10)

        # -----------------------------------------------------
        # Swap behavior
        # -----------------------------------------------------

        if action == "swap_weapon":
            if not trusted_enemy_visible:
                reward += self.add_penalty("swap_without_enemy", -0.35)

            if weapon in ["pistol", "shotgun", "chaingun", "plasma"] and trusted_enemy_visible:
                reward += self.add_penalty("swapped_away_from_good_weapon", -0.45)

            if self.consecutive_swap_steps > 1:
                reward += self.add_penalty("repeated_weapon_swap", -0.60)

        return reward

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
    
    def goal_progress_reward(self, director_result):
        """
        Small dense reward for moving closer to the current director target.

        This helps the agent understand:
        - closer to goal = good
        - small progress matters
        - route movement should continue
        """

        if director_result is None:
            return 0.0

        distance_delta = float(director_result.get("distance_delta", 0.0) or 0.0)

        reward = 0.0

        if distance_delta > 1.0:
            reward += 0.03
            self.reward_manager.add("small_goal_progress", 0.03)

        if distance_delta > 8.0:
            reward += 0.07
            self.reward_manager.add("medium_goal_progress", 0.07)

        if distance_delta > 20.0:
            reward += 0.10
            self.reward_manager.add("large_goal_progress", 0.10)

        if distance_delta < -12.0:
            reward -= 0.08
            self.reward_manager.add("moving_away_from_goal", -0.08)

        return reward

    def exploration_assist_action(self, action, enemy_visible, distance_moved=None, motion=None):
        """
        Advisory-only exploration.

        Keep this helper available for logging/future reward shaping,
        but never replace PPO movement choices during route learning.
        """
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
    
    def sensory_emergency_action(
        self,
        action,
        scene_label,
        scene_confidence,
        wall_info,
        stuck_counter,
        wall_contact_steps,
        motion=None,
        distance_moved=None,
    ):
        """
        Pre-action sensory emergency controller.

        This should only activate during real movement danger:
        - strong front wall
        - repeated stuck counter
        - repeated wall contact
        - boxed-in view
        - low movement after attempting movement

        It must NOT override normal navigation.
        """

        front_ratio = float(wall_info.get("front_ratio", 0.0))
        left_ratio = float(wall_info.get("left_ratio", 0.0))
        right_ratio = float(wall_info.get("right_ratio", 0.0))

        front_blocked = (
            wall_info.get("front_wall", False)
            or front_ratio >= 0.62
            or (
                scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall"]
                and scene_confidence >= 0.85
                and front_ratio >= 0.45
            )
        )

        boxed_in = (
            front_ratio >= 0.80
            and left_ratio >= 0.80
            and right_ratio >= 0.80
        )

        low_motion = (
            motion is not None
            and distance_moved is not None
            and action in ["move_forward", "move_backward", "strafe_left", "strafe_right"]
            and motion < 1.5
            and distance_moved < 1.0
        )

        serious_stuck = (
            stuck_counter >= 8
            or wall_contact_steps >= 3
            or boxed_in
            or low_motion
            or (
                action == "move_forward"
                and front_blocked
            )
        )

        # Critical rule:
        # Do not override normal movement just because the sensory model has a route-area label.
        if not serious_stuck:
            self.sensory_emergency_active = False
            self.sensory_emergency_steps = 0
            return None

        self.sensory_emergency_active = True
        self.sensory_emergency_steps += 1

        if boxed_in:
            cycle = self.sensory_emergency_steps % 8

            if cycle in [1, 2, 3]:
                return "turn_left"
            elif cycle in [4, 5]:
                return "move_backward"
            elif cycle == 6:
                return "strafe_left"
            else:
                return "turn_right"

        if front_blocked:
            if left_ratio < right_ratio and left_ratio < 0.65:
                return "turn_left"

            if right_ratio < left_ratio and right_ratio < 0.65:
                return "turn_right"

            cycle = self.sensory_emergency_steps % 6

            if cycle in [1, 2]:
                return "move_backward"
            elif cycle in [3, 4]:
                return "turn_left"
            else:
                return "strafe_right"

        cycle = self.sensory_emergency_steps % 8

        if cycle in [1, 2]:
            return "move_backward"
        elif cycle in [3, 4]:
            return "turn_right"
        elif cycle == 5:
            return "move_forward"
        elif cycle == 6:
            return "strafe_right"
        else:
            return "turn_left"

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
        if bubble["level"] == "orange":
            if action == "move_forward" and bubble["direction"] == "front" and bubble["front"] >= 0.60:
                return "move_backward"

            if action == "strafe_left" and bubble["direction"] == "left" and bubble["left"] >= 0.75:
                return "strafe_right"

            if action == "strafe_right" and bubble["direction"] == "right" and bubble["right"] >= 0.75:
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

        x = game_state.get("x")
        y = game_state.get("y")

        if x is not None and y is not None:
            x = float(x)
            y = float(y)

            # If the agent is near the useful secret/side route,
            # do not let goal steering override every movement choice.
            if -280.0 <= x <= -140.0 and 80.0 <= y <= 220.0:
                return action

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
    
    def wall_sensor_reward(self, wall_sensor_state, action, distance_moved, motion):
        """
        Reward shaping for visible wall/obstacle avoidance.

        This does not force actions. It teaches the policy that pushing into
        blocked geometry is bad and escaping from pressure is good.
        """

        if not wall_sensor_state:
            return 0.0

        reward = 0.0

        front_blocked = wall_sensor_state.get("front_blocked", False)
        near_wall = wall_sensor_state.get("near_wall", False)
        obstacle_pressure = wall_sensor_state.get("obstacle_pressure", 0.0)
        safest = wall_sensor_state.get("safest_escape_direction")

        movement_action = action in [
            "move_forward",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]

        if action == "move_forward" and front_blocked and distance_moved < 1.0:
            reward += self.add_penalty("wall_sensor_forward_blocked", -1.5)

        if movement_action and near_wall and distance_moved < 0.5 and motion < 1.0:
            reward += self.add_penalty("wall_sensor_no_progress_near_wall", -1.0)

        if near_wall and action in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
            reward += 0.15
            self.reward_manager.add("wall_sensor_escape_action", 0.15)

        if safest == "right" and action in ["turn_right", "strafe_right"]:
            reward += 0.08
            self.reward_manager.add("wall_sensor_escape_right", 0.08)

        if safest == "left" and action in ["turn_left", "strafe_left"]:
            reward += 0.08
            self.reward_manager.add("wall_sensor_escape_left", 0.08)

        if obstacle_pressure < 0.25 and action == "move_forward" and distance_moved > 3.0:
            reward += 0.05
            self.reward_manager.add("wall_sensor_open_forward", 0.05)

        return reward
    

    def tactical_weapon_action(
        self,
        action,
        game_state,
        object_result=None,
        wall_info=None,
    ):
        """
        Situational weapon swapping.

        This prevents blind swap_weapon spam while still allowing:
        - berserk/fist for very close enemies
        - guns for medium/far enemies
        - rockets only for groups at safe distance
        """

        if self.curriculum_stage < 3:
            if action == "swap_weapon":
                self.last_forced_weapon_reason = "swap_blocked_before_combat_stage"
                return "move_forward"
            return action

        if action != "swap_weapon":
            return action

        if self._step_count - self.last_weapon_swap_step < self.weapon_swap_cooldown:
            self.last_forced_weapon_reason = "swap_cooldown"
            return "move_backward"

        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        enemy_distance_tiles = self.infer_enemy_distance_tiles(game_state, object_result)
        enemy_count = self.infer_enemy_count(game_state, object_result)

        current_weapon = str(
            game_state.get("weapon", self.current_weapon_name)
        ).lower()

        front_ratio = 0.0
        if wall_info:
            front_ratio = float(wall_info.get("front_ratio", 0.0))

        rocket_unsafe = (
            front_ratio >= 0.45
            or (
                enemy_distance_tiles is not None
                and enemy_distance_tiles < 4.0
            )
        )

        if not enemy_visible:
            self.last_forced_weapon_reason = "swap_blocked_no_enemy"
            return "move_forward"

        if self.has_berserk and enemy_distance_tiles is not None:
            if enemy_distance_tiles <= 1.5 and enemy_centered:
                self.last_weapon_swap_step = self._step_count
                self.last_forced_weapon_reason = "berserk_close_enemy"
                return "swap_weapon"

            if current_weapon in ["fist", "chainsaw", "riptor"] and enemy_distance_tiles > 2.0:
                self.last_weapon_swap_step = self._step_count
                self.last_forced_weapon_reason = "leave_melee_range"
                return "swap_weapon"

        if enemy_count >= 2 and not rocket_unsafe:
            self.last_weapon_swap_step = self._step_count
            self.last_forced_weapon_reason = "group_enemy_safe_rocket"
            return "swap_weapon"

        self.last_forced_weapon_reason = "swap_not_contextual"
        return "shoot" if enemy_centered else "move_backward"

    def sanitize_action(self, action, game_state, enemy_visible):
        """
        Final action gate for the current curriculum stage.

        This blocks illegal/impossible actions, but it does NOT create new
        combat behavior. In particular, it must not convert swap_weapon or
        no-ammo cases back into shoot.
        """
        config = self.get_stage_config()

        ammo = int(game_state.get("ammo", 0) or 0)

        if ammo <= 0 and action == "shoot":
            before = action

            if self.stuck_counter >= 3:
                action = "move_backward"
            elif self._step_count % 2 == 0:
                action = "strafe_left"
            else:
                action = "strafe_right"

            print(f"[override] no_ammo: {before} -> {action}")

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

            classifier_enemy = (
                game_state.get("scene_label") == "enemy"
                and float(game_state.get("scene_confidence", 0.0)) >= 0.70
            )

            object_enemy = False
            object_vision = game_state.get("object_vision")

            if isinstance(object_vision, dict):
                scores = object_vision.get("scores", {}) or {}
                present = set(object_vision.get("present", []) or [])

                object_enemy = (
                    "enemy_visible" in present
                    or float(scores.get("enemy_visible", 0.0)) >= 0.55
                )

            trusted_enemy_visible = enemy_visible or classifier_enemy or object_enemy

            if config.get("require_enemy_visible_to_shoot", False) and not trusted_enemy_visible:
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
