import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import time
import torch

from pynput import keyboard

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
from recording.record_human_play import HumanRecorder
from recording.clip_generator import ClipGenerator
from recording.smart_clip_generator import SmartClipGenerator
from recording.record_player_session import PlayerRecorder
from recording.event_recorder import EventRecorder
from curriculum.curriculum_manager import CurriculumManager
from observation.frame_processor import FrameProcessor


DOOM_BINARY = "/home/steven/Downloads/doomretro-master/build/doomretro"
DOOM_IWAD = "/usr/share/games/doom/freedoom1.wad"

class DoomEnv(gym.Env):

    def __init__(self, launch_doom=True, record=True):

        super(DoomEnv, self).__init__()

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

        self.controller = DoomController()
        self.observer = ObservationBuilder()
        self.frame_stack = FrameStack(stack_size=3)
        self.exploration_memory = ExplorationMemory(map_size=512)
        self.reward_manager = RewardManager()
        self.actions = ActionSpace().ACTIONS
        self.preference_model = PreferenceModel()
        self.helper = HelperBot()
        self.record = record
        self.reward_overlay = RewardDebugOverlay() if RewardDebugOverlay is not None else None
        self.stuck_counter = 0
        self.curriculum_stage = 0
        self.stage = 0
        self.rotation_state = DoomController.RotationState()
        self.frame_processor = FrameProcessor()

        # FIX #4: CurriculumManager is now synced to self.curriculum_stage.
        # get_stage_name() takes curriculum_stage as an argument instead of
        # maintaining its own independent counter.
        self.curriculum = CurriculumManager()

        self.curriculum_rewards = []
        self.curriculum_thresholds = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.1,
            4: 0.2,
            5: 0.0,
            6: 0.2,
            7: 0.3,
        }
        self.max_stage = 7
        self.last_player_position = None
        self.distance_traveled = 0.0
        self.visited_tiles = set()
        self.enemy_visible_steps = 0
        self.initial_distance = None
        self.closest_distance = None
        self.last_damage_source = None

        # Stage-specific behavior counters
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
        self.visited_areas = set()
        self.swap_weapon_count = 0


        self.event_recorder = EventRecorder()
        self.smart_clipper = SmartClipGenerator()

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

        pref_path = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "preference_model.pt"
        )
        if os.path.exists(pref_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.preference_model.load_state_dict(
                torch.load(pref_path, map_location=device)
            )
            self.preference_model.eval()
            self._use_preference = True
        else:
            print("No preference_model.pt found — skipping preference reward.")
            self._use_preference = False

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8
        )

        self.previous_health = None
        self._step_count = 0
        self._max_episode_steps = 1000
        self._previous_frame = None
        self.last_shot_step = -100
        self.enemy_visible_steps = 0
        self.last_area_signature = None
        frame = self.observer.get_frame()

        if self.record:
            self._kb_listener = keyboard.Listener(
                on_press=self.recorder.on_press,
                on_release=self.recorder.on_release,
            )
            self._kb_listener.start()
        else:
            self._kb_listener = None

        self.last_area_signature = None

    def get_stage_config(self):
        return {
            0: {
                "name": "movement",
                "allow_shoot": False,
                "enemy_enabled": False,
                "strafe_left": True,
                "strafe_right": True,
                "goal": "move"
            },
            1: {
                "name": "game_basics",
                "allow_shoot": False,
                "enemy_enabled": False,
                "strafe_left": True,
                "strafe_right": True,
                "goal": "explore_and_pick_items"
            },
            2: {
                "name": "avoid_getting_stuck",
                "allow_shoot": False,
                "enemy_enabled": False,
                "strafe_left": True,
                "strafe_right": True,
                "goal": "avoid_stuck"
            },
            3: {
                "name": "combat_basic",
                "visible": True,
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "dodge_enemies": True,
                "track_enemy": True,
                "enemy_enabled": True,
                "strafe_left": True,
                "strafe_right": True,
                "Swap_Weapon": True,
                "max_enemies": 1,
                "goal": "defeat_enemies"
            },
            4: {
                "name": "combat_full",
                "allow_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "strafe_left": True,
                "strafe_right": True,
                "Swap_Weapon": True,
                "max_enemies": min(10, self._step_count // 200),
                "goal": "defeat_enemies"
            },
            5: {
                "name": "key_doors",
                "allow_shoot": False,
                "require_enemy_visible_to_shoot": True,
                "strafe_left": True,
                "strafe_right": True,
                "Swap_Weapon": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "enemy_enabled": False,
                "goal": "open_doors"
            },
            6: {
                "name": "full_game",
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "dodge_enemies": True,
                "track_enemy": True,
                "strafe_left": True,
                "strafe_right": True,
                "Swap_Weapon": True,
                "max_enemies": 1000,
                "goal": "defeat_enemies"
            },
            7: {
                "name": "complete_level",
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "enemy_enabled": True,
                "strafe_left": True,
                "strafe_right": True,
                "Swap_Weapon": True,
                "max_enemies": 5000,
                "goal": "finish_level"
            }
        }[self.curriculum_stage]
    
    def reset(self, seed=None, options=None):
        self._step_count = 0

        self.movement_count = 0
        self.exploration_count = 0
        self.pickup_count = 0
        self.continuous_movement_steps = 0
        self.door_interaction_count = 0
        self.track_enemy_count = 0
        self.dodge_enemies_count = 0
        self.key_item_count = 0
        self.enemy_engagement_count = 0
        self.valid_shot_count = 0
        self.enemy_visible_steps = 0
        self.stuck_counter = 0
        self.distance_traveled = 0.0
        self.visited_tiles.clear()
        self.last_player_position = self.observer.get_player_position()

        goal_pos = self.observer.get_goal_position()
        player_pos = self.observer.get_player_position()

        dist = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

        self.initial_distance = dist
        self.closest_distance = dist

        self.visited_areas.clear()

        self.reward_manager.reset()
        self.reward_manager.initial_distance = dist
        self.reward_manager.closest_distance = dist

        config = self.get_stage_config()
        print(f"[Curriculum] Stage {self.curriculum_stage}: {config['name']}")

        import subprocess
        wid = self.controller.window_id
        if wid:
            subprocess.call(
                ["xdotool", "windowfocus", "--sync", wid, "key", "Return"],
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.5)

        # Re-detect window position before every capture so stale coords from
        # a previous episode never cause XGetImage() to grab an invalid region.
        frame_cache.reset_monitor()
        frame_cache.invalidate()

        self.previous_health = None
        self._previous_frame = None

        frame = self.observer.build()
        observation = self.frame_stack.reset(frame)
        return observation, {}
    
    def get_allowed_actions(self):

        stage = self.curriculum_stage

        if stage == 0:
            return [
                "move_forward",
                "turn_left",
                "turn_right",
            ]

        elif stage == 1:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "use",
            ]

        elif stage == 2:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
            ]

        elif stage >= 3:
            return [
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
        elif stage == 4:
            return [
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
        elif stage == 5:
            return [
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
        elif stage == 6:
            return [
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
        elif stage == 7:
            return [
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

    # CORE STEP FLOW (STABLE VERSION)

def step(self, action_index):

    frame_cache.invalidate()

    reward = 0.0
    done = False

    # ---------------------------------------------------------
    # VALID ACTION MASKING
    # ---------------------------------------------------------

    if action_index not in self.get_valid_actions():
        reward -= 0.2
        return obs, reward, False, False, {}
    # ---------------------------------------------------------
    # ACTION LOOKUP
    # ---------------------------------------------------------

    action = self.actions[action_index]

    # ---------------------------------------------------------
    # EXECUTE ACTION
    # ---------------------------------------------------------

    self.perform_action(action)

    # ---------------------------------------------------------
    # OBSERVE FRAME
    # ---------------------------------------------------------

    frame = self.observer.get_frame()

    observation_tensor, motion, cx, cy = (
        self.frame_processor.extract(frame)
    )

    # ---------------------------------------------------------
    # ENEMY DETECTION
    # ---------------------------------------------------------

    enemy_visible = (
        self.reward_manager.enemy_detector
        .detect_enemy_presence(frame)
    )

    if enemy_visible:
        self.enemy_visible_steps += 1

    # ---------------------------------------------------------
    # ENVIRONMENT ANALYSIS
    # ---------------------------------------------------------

    floor_green_ratio = (
        self.frame_processor.floor_green_ratio(frame)
    )

    red_ratio = (
        self.frame_processor.red_flash_ratio(frame)
    )

    # ---------------------------------------------------------
    # STUCK DETECTION
    # ---------------------------------------------------------

    if motion < 1.5:

        self.stuck_counter += 1

    else:

        if self.stuck_counter > 10:
            reward += 2.0

        self.stuck_counter = 0

    # punish wall scraping
    if motion < 1.0 and action == "move_forward":

        reward -= 0.15

    # reward escape behavior
    if (
        self.stuck_counter > 10
        and action in [
            "turn_left",
            "turn_right",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]
    ):

        reward += 0.1

    # severe stuck punishment
    if self.stuck_counter > 25:

        reward -= 1.0

    # ---------------------------------------------------------
    # GAME STATE
    # ---------------------------------------------------------

    game_state = self.observer.get_game_state()

    game_state["enemy_visible"] = enemy_visible
    game_state["motion"] = motion

    # ---------------------------------------------------------
    # DISTANCE TRACKING
    # ---------------------------------------------------------

    distance_moved = 0.0

    if game_state["shared_state_available"]:

        player_x = game_state["x"]
        player_y = game_state["y"]

        current_player_pos = (
            player_x,
            player_y
        )

        if self.last_player_position is not None:

            distance_moved = float(
                np.linalg.norm(
                    np.array(current_player_pos)
                    - np.array(self.last_player_position)
                )
            )

        self.last_player_position = current_player_pos

        self.distance_traveled += distance_moved

        # exploration reward
        tile = (
            int(player_x // 64),
            int(player_y // 64)
        )

        if tile not in self.visited_tiles:

            self.visited_tiles.add(tile)

            reward += 0.05

            self.exploration_count += 1

        # novelty memory
        self.exploration_memory.visit(
            player_x,
            player_y
        )

        novelty_reward = (
            self.exploration_memory.get_novelty(
                player_x,
                player_y
            ) * 0.3
        )

        self.reward_manager.add(
            "exploration_novelty",
            novelty_reward
        )

    # ---------------------------------------------------------
    # COMBAT REWARDS
    # ---------------------------------------------------------

    if action == "shoot":

        if enemy_visible:

            reward += 1.0

            self.enemy_engagement_count += 1

        else:

            reward -= 1.5

        if self.reward_manager.enemy_in_crosshair(frame):

            reward += 2.0

            self.valid_shot_count += 1

    # ---------------------------------------------------------
    # AIMING REWARD
    # ---------------------------------------------------------

    if (
        enemy_visible
        and self.reward_manager.enemy_in_crosshair(frame)
    ):

        reward += 0.3

    # ---------------------------------------------------------
    # ITEM REWARDS
    # ---------------------------------------------------------

    if (
        game_state["health_delta"] > 0
        or game_state["ammo_delta"] > 0
    ):

        reward += 0.5

        self.pickup_count += 1

    # ---------------------------------------------------------
    # ACID / BARREL DAMAGE DETECTION
    # ---------------------------------------------------------

    self.reward_manager.detect_acid_damage(
        game_state["health_delta"],
        distance_moved,
        floor_green_ratio
    )

    self.reward_manager.detect_barrel_damage(
        game_state["health_delta"],
        red_ratio
    )

    # ---------------------------------------------------------
    # MOVEMENT REWARD
    # ---------------------------------------------------------

    if distance_moved > 5.0:

        reward += 0.03

        self.movement_count += 1

    # ---------------------------------------------------------
    # STAGNATION PENALTY
    # ---------------------------------------------------------

    self.reward_manager.update_stagnation_penalty(
        self.stuck_counter
    )

    reward += self.reward_manager.get_reward()

    # ---------------------------------------------------------
    # LEVEL COMPLETION
    # ---------------------------------------------------------

    if game_state.get("level_complete", False):

        reward += 100.0

        self.level_completion_count += 1

        done = True

    # ---------------------------------------------------------
    # PPO TIME LIMIT
    # ---------------------------------------------------------

    self._step_count += 1

    truncated = (
        self._step_count >= self._max_episode_steps
    )

    # ---------------------------------------------------------
    # BUILD OBSERVATION
    # ---------------------------------------------------------

    processed_frame = self.observer.build()

    observation = self.frame_stack.add_frame(
        processed_frame
    )

    # ---------------------------------------------------------
    # RECORDING
    # ---------------------------------------------------------

    if self.record:

        self.logger.log(
            frame,
            action,
            game_state["health"],
            game_state["ammo"]
        )

        self.traj_logger.record(
            frame,
            action,
            reward
        )

        self.recorder.record_frame(frame)

        self.smart_clipper.observe(
            frame,
            action
        )

    # ---------------------------------------------------------
    # CURRICULUM
    # ---------------------------------------------------------

    self.curriculum_rewards.append(reward)

    if len(self.curriculum_rewards) > 100:

        self.curriculum_rewards.pop(0)

    self.update_curriculum()

    return (
        observation,
        reward,
        done,
        truncated,
        {}
    )
def perform_action(self, action):

    allowed = self.get_allowed_actions()

    if action not in allowed:
        return

    # STOP CONTINUOUS KEYS
    self.controller.stop_forward_backward()
    self.controller.stop_turn()

    # ---------------------------------------------------
    # MOVEMENT
    # ---------------------------------------------------

    if action == "move_forward":

        self.controller.start_move_forward()

    elif action == "move_backward":

        self.controller.start_move_backward()

    # ---------------------------------------------------
    # TURNING
    # ---------------------------------------------------

    elif action == "turn_left":

        self.controller.start_turn_left()

    elif action == "turn_right":

        self.controller.start_turn_right()

    # ---------------------------------------------------
    # STRAFING
    # ---------------------------------------------------

    elif action == "strafe_left":

        self.controller.strafe_left()

    elif action == "strafe_right":

        self.controller.strafe_right()

    # ---------------------------------------------------
    # COMBAT
    # ---------------------------------------------------

    elif action == "shoot":

        self.controller.shoot()

    # ---------------------------------------------------
    # INTERACTION
    # ---------------------------------------------------

    elif action == "use":

        self.controller.use()

    elif action == "swap_weapon":

        self.controller.swap_weapon()
