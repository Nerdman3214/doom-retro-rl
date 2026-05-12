import sys
import os

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


DOOM_BINARY = "/home/steven/Downloads/doomretro-master/build/doomretro"
DOOM_IWAD = "/usr/share/games/doom/freedoom2.wad"


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

        # Core systems
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
        self.prev_enemy_visible = False
        self.curriculum_log_interval = 50

        # Recording/debug systems
        self.record = record
        self.reward_overlay = RewardDebugOverlay() if RewardDebugOverlay is not None else None
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

        if self.record and keyboard is not None and self.recorder is not None:
            self._kb_listener = keyboard.Listener(
                on_press=self.recorder.on_press,
                on_release=self.recorder.on_release,
            )
            self._kb_listener.start()
        else:
            self._kb_listener = None

        # RL spaces
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8,
        )

        # Curriculum
        self.curriculum_stage = 0
        self.max_stage = 7
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

        # Episode state
        self._step_count = 0
        self._max_episode_steps = 1000
        self.stuck_counter = 0
        self.previous_health = None
        self._previous_frame = None
        self.last_shot_step = -100
        self.last_player_position = None
        self.last_area_signature = None
        self.initial_distance = None
        self.closest_distance = None
        self.distance_traveled = 0.0
        self.visited_tiles = set()
        self.visited_areas = set()
        self.visual_area_counter = 0
        self.last_visual_signature = None
        self.visual_area_steps = 0

        # Behavior counters
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

        self._use_preference = False
        self._load_preference_model()

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
                "name": "movement",
                "allow_shoot": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
            },
            1: {
                "name": "game_basics",
                "allow_shoot": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
            },
            2: {
                "name": "avoid_getting_stuck",
                "allow_shoot": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
            },
            3: {
                "name": "combat_basic",
                "visible": True,
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "dodge_enemies": True,
                "track_enemy": True,
                "enemy_enabled": True,
                "cautious_explore": True,
                "allow_goal_progress": True,
                "allow_door_use": True,
                "allow_elevator_progress": True,
                "max_enemies": 1,
                "goal": "cautious_combat_progression",
            },
            4: {
                "name": "combat_full",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "weapon_awareness": True,
                "ammo_awareness": True,
                "melee_awareness": True,
                "goal": "combat_with_resources",
            },
            5: {
                "name": "key_doors",
                "allow_shoot": False,
                "require_enemy_visible_to_shoot": False,
                "track_enemy": False,
                "dodge_enemies": False,
            },
            6: {
                "name": "full_game",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "weapon_awareness": True,
                "ammo_awareness": True,
                "melee_awareness": True,
            },
            7: {
                "name": "complete_level",
                "allow_shoot": True,
                "allow_melee": True,
                "require_enemy_visible_to_shoot": True,
                "track_enemy": True,
                "dodge_enemies": True,
                "weapon_awareness": True,
                "ammo_awareness": True,
                "melee_awareness": True,
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
                "use",
            ]

        if stage == 2:
            return [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "strafe_left",
                "strafe_right",
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
        self.stuck_counter = 0
        self.previous_health = None
        self._previous_frame = None
        self.last_shot_step = -100
        self.distance_traveled = 0.0
        self.visited_tiles.clear()
        self.visited_areas.clear()
        self.last_area_signature = None

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

        self.reward_manager.reset()

        if hasattr(self.observer, "reset_tracking"):
            self.observer.reset_tracking()

        if hasattr(self.reward_manager, "enemy_detector"):
            if hasattr(self.reward_manager.enemy_detector, "reset"):
                self.reward_manager.enemy_detector.reset()

        frame_cache.reset_monitor()
        frame_cache.invalidate()

        wid = self.controller.window_id
        if wid:
            subprocess.call(
                ["xdotool", "windowfocus", "--sync", wid],
                stderr=subprocess.DEVNULL,
            )

            time.sleep(0.2)

            # Press Return more than once because Doom death/intermission/menu states
            # may need an input to respawn or dismiss the screen.
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

        # Build safe observation before any possible early return.
        safe_frame = self.observer.build()
        safe_observation = self.frame_stack.add_frame(safe_frame)

        # Reject invalid curriculum actions early.
        if action_index not in self.get_valid_actions():
            reward -= 0.2
            return safe_observation, float(reward), terminated, truncated, info

        # Convert action index into action name.
        action = self.actions[action_index]
        config = self.get_stage_config()

        # Read current frame/state before executing the action.
        pre_frame = self.observer.get_frame()
        pre_game_state = self.observer.get_game_state()

        pre_enemy_visible = self.reward_manager.enemy_detector.detect_enemy_presence(pre_frame)
        pre_enemy_centered = self.reward_manager.enemy_in_crosshair(pre_frame)

        pre_game_state["curriculum_stage"] = self.curriculum_stage
        pre_game_state["enemy_visible"] = pre_enemy_visible
        pre_game_state["enemy_centered"] = pre_enemy_centered
        pre_game_state["action"] = action

        # Convert unsafe actions into safer actions.
        action = self.sanitize_action(action, pre_game_state, pre_enemy_visible)

        # Aim assist: if an enemy is visible, turn the shortest direction
        # or shoot if already centered.
        action = self.aim_assist_action(
            action=action,
            frame=pre_frame,
            enemy_visible=pre_enemy_visible,
            enemy_centered=pre_enemy_centered,
            ammo=pre_game_state.get("ammo", 0),
        )

        # Helper alignment reward. Keep this small so it does not dominate PPO.
        helper_action = self.helper.get_action(pre_game_state)

        if action == helper_action:
            reward += 0.05
        else:
            reward -= 0.01

        # Execute action.
        self.perform_action(action)

        # Observe world after action.
        raw_frame = self.observer.get_frame()
        _, motion, _, _ = self.frame_processor.extract(raw_frame)

        enemy_visible = self.reward_manager.enemy_detector.detect_enemy_presence(raw_frame)
        enemy_centered = self.reward_manager.enemy_in_crosshair(raw_frame)

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
        )
        game_state["distance_moved"] = distance_moved

        current_weapon = str(game_state.get("weapon", "")).lower()
        ammo = game_state.get("ammo", 0)
        ammo_delta = game_state.get("ammo_delta", 0)
        weapon_delta = game_state.get("weapon_delta", 0)
        kill_delta = game_state.get("kill_delta", 0)

        enemy_close = (
            enemy_visible
            and enemy_centered
            and motion > 0.5
        )

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
            reward -= 0.15

        if self.stuck_counter > 10 and action in [
            "turn_left",
            "turn_right",
            "move_backward",
            "strafe_left",
            "strafe_right",
        ]:
            reward += 0.1

        if self.stuck_counter > 25:
            reward -= 1.0

        if distance_moved > 5.0:
            reward += 0.03
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
        # Stage 3 cautious combat progression
        # -----------------------------------------------------

        if self.curriculum_stage == 3:
            movement_actions = [
                "move_forward",
                "move_backward",
                "strafe_left",
                "strafe_right",
            ]

            # Keep progressing through the map while searching for enemies.
            if not enemy_visible and action == "move_forward":
                reward += 0.03

            if not enemy_visible and action == "use":
                reward += 0.08

            if distance_moved > 5.0:
                reward += 0.04

            if len(self.visited_tiles) > 0 and action in movement_actions:
                reward += 0.01

            # Discourage endless scanning when nothing is visible.
            if action in ["turn_left", "turn_right"] and not enemy_visible:
                reward -= 0.02

            # But allow turning once an enemy is visible because aiming may require it.
            if action in ["turn_left", "turn_right"] and enemy_visible:
                reward += 0.05

            # Enemy discovery reward should only happen on the transition:
            # not visible -> visible.
            if enemy_visible and not self.prev_enemy_visible:
                reward += 2.0
                self.reward_manager.add("enemy_discovered", 2.0)

            if enemy_visible:
                reward += 0.50

            if enemy_visible and enemy_centered:
                reward += 1.00
                self.track_enemy_count += 1

            # Correct shooting.
            if action == "shoot" and enemy_visible:
                reward += 1.00

                if enemy_centered:
                    reward += 2.00
                    self.enemy_engagement_count += 1
                    self.valid_shot_count += 1

            # Bad shooting.
            if action == "shoot" and not enemy_visible:
                reward -= 1.00

        # -----------------------------------------------------
        # Generic combat rewards for stages after cautious combat
        # -----------------------------------------------------

        elif self.curriculum_stage >= 4:
            if action == "shoot":
                if not config.get("allow_shoot", False):
                    reward -= 1.0

                elif ammo <= 0:
                    reward -= 1.0
                    self.reward_manager.add("shoot_no_ammo", -1.0)

                elif config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
                    reward -= 1.0
                    self.reward_manager.add("shoot_without_visible_enemy", -1.0)

                else:
                    if enemy_visible:
                        reward += 0.6
                        self.enemy_engagement_count += 1

                    if enemy_visible and enemy_centered:
                        reward += 1.2
                        self.valid_shot_count += 1
                        self.reward_manager.add("accurate_shot", 1.2)

                    if enemy_visible and not enemy_centered:
                        reward += 0.15

                    if ammo <= 5:
                        reward -= 0.2
                        self.reward_manager.add("low_ammo_shot_pressure", -0.2)
        # -----------------------------------------------------
        # Melee combat rewards
        # -----------------------------------------------------

        if action == "melee_attack":
            self.melee_attack_count += 1

            if not config.get("allow_melee", False):
                reward -= 1.0

            elif not enemy_visible:
                reward -= 0.4
                self.reward_manager.add("melee_without_enemy", -0.4)

            elif enemy_visible and enemy_centered:
                reward += 0.8
                self.reward_manager.add("melee_enemy_centered", 0.8)

                if enemy_close:
                    reward += 1.5
                    self.melee_close_bonus_count += 1
                    self.reward_manager.add("melee_close_range", 1.5)

            else:
                reward -= 0.2

        # -----------------------------------------------------
        # Use / weapon rewards
        # -----------------------------------------------------

        if action == "use":
            # Only count use as meaningful if the agent is not just spamming it.
            meaningful_use = (
                distance_moved > 2.0
                or motion > 2.0
                or self.stuck_counter < 5
            )

            if meaningful_use:
                self.door_interaction_count += 1

                if self.curriculum_stage == 3:
                    reward += 0.05

                if self.curriculum_stage == 5:
                    reward += 1.0
            else:
                reward -= 0.1
                self.reward_manager.add("use_spam_penalty", -0.1)

        if action == "swap_weapon":
            self.swap_weapon_count += 1
            ammo = game_state.get("ammo", 0)

            if ammo <= 2:
                reward += 0.2
            else:
                reward -= 0.5

        # -----------------------------------------------------
        # Weapon quality awareness
        # -----------------------------------------------------

        if self.curriculum_stage >= 4:
            using_weak_weapon = (
                "pistol" in current_weapon
                or current_weapon == ""
            )

            if using_weak_weapon and self._step_count > 300:
                reward -= 0.02
                self.reward_manager.add("weak_weapon_pressure", -0.02)

            if self.weapon_pickup_count > 0:
                reward += 0.05

        # -----------------------------------------------------
        # Stage 7 final-game progression rewards
        # -----------------------------------------------------

        if self.curriculum_stage >= 7:
            # Keep moving through the level.
            if distance_moved > 5.0:
                reward += 0.05

            # Encourage using doors/switches/elevators.
            if action == "use":
                reward += 0.15

            # Discourage endless turning when no enemy is visible.
            if action in ["turn_left", "turn_right"] and not enemy_visible:
                reward -= 0.02

            # Reward survival lightly, but do not let survival farming dominate.
            if game_state.get("health", 100) > 0:
                reward += 0.005

            # Strongly reward actual completion.
            if game_state.get("level_complete", False):
                reward += 100.0

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

        # Update enemy transition memory after all enemy-related rewards.
        self.prev_enemy_visible = enemy_visible

        # -----------------------------------------------------
        # Completion / death / termination
        # -----------------------------------------------------

        health = game_state.get("health", 100)

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

        # -----------------------------------------------------
        # Final observation
        # -----------------------------------------------------

        processed_frame = self.observer.build()
        observation = self.frame_stack.add_frame(processed_frame)

        # -----------------------------------------------------
        # Recording
        # -----------------------------------------------------

        if self.record:
            self._record_step(raw_frame, action, reward, game_state)

        # -----------------------------------------------------
        # Curriculum update
        # -----------------------------------------------------

        self.curriculum_rewards.append(reward)

        if len(self.curriculum_rewards) > 100:
            self.curriculum_rewards.pop(0)

        self.update_curriculum()

        return observation, float(reward), terminated, truncated, info

    # ---------------------------------------------------------
    # Action execution
    # ---------------------------------------------------------

    def perform_action(self, action):
        allowed = self.get_allowed_actions()

        if action not in allowed:
            return

        self.controller.release_all()

        if action == "move_forward":
            self.controller.start_move_forward()

        elif action == "move_backward":
            self.controller.start_move_backward()

        elif action == "turn_left":
            if hasattr(self.controller, "quick_turn_left"):
                self.controller.quick_turn_left()
            else:
                self.controller.start_turn_left()

        elif action == "turn_right":
            if hasattr(self.controller, "quick_turn_right"):
                self.controller.quick_turn_right()
            else:
                self.controller.start_turn_right()

        elif action == "strafe_left":
            self.controller.start_strafe_left()

        elif action == "strafe_right":
            self.controller.start_strafe_right()

        elif action == "shoot":
            self.controller.shoot()

        elif action == "melee_attack":
            self.controller.shoot()

        elif action == "use":
            self.controller.use()

        elif action == "swap_weapon":
            self.controller.swap_weapon()

        

    # ---------------------------------------------------------
    # Tracking / reward helpers
    # ---------------------------------------------------------

    def _visual_area_signature(self, frame):
        """
        Fallback exploration signature when shared-memory x/y is unavailable.

        This does not know the real map position. It only detects whether
        the visual scene has changed enough to count as approximate progress.
        """
        if frame is None or frame.size == 0:
            return None

        small = frame[::8, ::8, :].astype(np.int16)

        # Quantize colors to reduce noise.
        quantized = small // 32

        return hash(quantized.tobytes())

    def _update_position_tracking(self, game_state, frame=None, motion=0.0):
        distance_moved = 0.0

        if not game_state.get("shared_state_available", False):
            # Fallback: use screen motion as approximate movement.
            if motion > 2.0:
                distance_moved = float(motion)
                self.distance_traveled += distance_moved

            signature = self._visual_area_signature(frame)

            if signature is not None and signature != self.last_visual_signature:
                self.last_visual_signature = signature
                self.visual_area_steps += 1

                # Every few meaningful visual changes count as a pseudo-tile.
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
            behavior_ready = self.movement_count >= 20 and self.distance_traveled >= 150.0
            if should_log:
                print(
                    f"  [Stage 0] moves: {self.movement_count}/20 | "
                    f"distance: {self.distance_traveled:.1f}/150 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )
            

        elif self.curriculum_stage == 1:
            behavior_ready = len(self.visited_tiles) >= 5 and self.pickup_count >= 1

            if should_log:
                print(
                    f"  [Stage 1] tiles: {len(self.visited_tiles)}/5 | "
                    f"pickups: {self.pickup_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )
            

        elif self.curriculum_stage == 2:
            behavior_ready = self.stuck_counter == 0 and self.movement_count >= 30
            if should_log:
                print(
                    f"  [Stage 2] moves: {self.movement_count}/30 | "
                    f"stuck: {self.stuck_counter}"
                )

        elif self.curriculum_stage == 3:
            behavior_ready = (
                (
                    self.enemy_engagement_count >= 5
                    and self.enemy_visible_steps >= 5
                )
                or
                (
                    len(self.visited_tiles) >= 8
                    and self.door_interaction_count >= 1
                )
            )
            if should_log:
                print(
                    f"  [Stage 3] shots: {self.enemy_engagement_count}/5 | "
                    f"visible: {self.enemy_visible_steps}/5 | "
                    f"tiles: {len(self.visited_tiles)}/8 | "
                    f"doors/use: {self.door_interaction_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 4:
            combat_hits = self.valid_shot_count + self.melee_close_bonus_count

            behavior_ready = (
                (
                    combat_hits >= 5
                    and self.enemy_visible_steps >= 8
                )
                or
                (
                    self.enemy_kill_count >= 1
                )
                or
                (
                    self.ammo_pickup_count >= 1
                    and self.weapon_pickup_count >= 1
                )
            )
            if should_log:
                print(
                    f"  [Stage 4] hits/melee: {combat_hits}/5 | "
                    f"visible: {self.enemy_visible_steps}/8 | "
                    f"kills: {self.enemy_kill_count}/1 | "
                    f"ammo: {self.ammo_pickup_count}/1 | "
                    f"weapons: {self.weapon_pickup_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 5:
            behavior_ready = self.door_interaction_count >= 1 and self.key_item_count >= 1
            if should_log:
                print(
                    f"  [Stage 5] doors: {self.door_interaction_count}/1 | "
                    f"keys/items: {self.key_item_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage == 6:
            combat_hits = self.valid_shot_count + self.melee_close_bonus_count

            behavior_ready = (
                (
                    len(self.visited_tiles) >= 12
                    and self.distance_traveled >= 500.0
                    and self.pickup_count >= 1
                )
                or
                (
                    len(self.visited_tiles) >= 10
                    and self.door_interaction_count >= 2
                )
                or
                (
                    self.level_completion_count >= 1
                )
                or
                (
                    combat_hits >= 3
                    and self.enemy_visible_steps >= 5
                )
            )
            if should_log:
                print(
                    f"  [Stage 6] tiles: {len(self.visited_tiles)}/12 | "
                    f"distance: {self.distance_traveled:.1f}/500 | "
                    f"doors: {self.door_interaction_count}/2 | "
                    f"pickups: {self.pickup_count}/1 | "
                    f"hits/melee: {combat_hits}/3 | "
                    f"visible: {self.enemy_visible_steps}/5 | "
                    f"complete: {self.level_completion_count}/1 | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        elif self.curriculum_stage >= 7:
            behavior_ready = True
            if should_log:
                print(
                    f"  [Stage 7] final/full game | "
                    f"tiles: {len(self.visited_tiles)} | "
                    f"distance: {self.distance_traveled:.1f} | "
                    f"doors: {self.door_interaction_count} | "
                    f"pickups: {self.pickup_count} | "
                    f"hits: {self.valid_shot_count + self.melee_close_bonus_count} | "
                    f"kills: {self.enemy_kill_count} | "
                    f"complete: {self.level_completion_count} | "
                    f"Reward: {avg_reward:.2f}/{threshold}"
                )

        if reward_ready and behavior_ready and self.curriculum_stage < self.max_stage:
            old_stage = self.curriculum_stage
            self.curriculum_stage += 1
            self.curriculum_rewards.clear()
            self._reset_stage_counters()

            print(
            f"[Curriculum] ADVANCING FROM STAGE "
            f"{old_stage} → {self.curriculum_stage}"
        )
            
    def _enemy_horizontal_error(self, frame):
        """
        Estimate whether the enemy is left or right of center.

        Returns:
            negative value = enemy is left
            positive value = enemy is right
            0.0 = centered or unknown
        """
        if frame is None or frame.size == 0:
            return 0.0

        heatmap = self.frame_processor.enemy_heatmap(frame)

        if heatmap is None or heatmap.size == 0:
            return 0.0

        ys, xs = np.where(heatmap > 0.5)

        if len(xs) == 0:
            return 0.0

        enemy_x = float(np.mean(xs))
        center_x = heatmap.shape[1] / 2.0

        # Normalize to roughly -1.0 to +1.0
        return float((enemy_x - center_x) / center_x)
    def aim_assist_action(self, action, frame, enemy_visible, enemy_centered, ammo):
        """
        Combat aim assist.

        This prevents the agent from turning the long way around.
        If the enemy is left, turn left.
        If the enemy is right, turn right.
        If centered and ammo exists, shoot.
        """
        if self.curriculum_stage < 3:
            return action

        if not enemy_visible:
            return action

        aim_error = self._enemy_horizontal_error(frame)

        # Enemy is centered enough: shoot.
        if enemy_centered or abs(aim_error) < 0.12:
            if ammo > 0:
                return "shoot"
            return "melee_attack"

        # Enemy is left of center.
        if aim_error < -0.12:
            return "turn_left"

        # Enemy is right of center.
        if aim_error > 0.12:
            return "turn_right"

        return action        

    def sanitize_action(self, action, game_state, enemy_visible):
        config = self.get_stage_config()

        ammo = game_state.get("ammo", 0)
        current_weapon = str(game_state.get("weapon", "")).lower()

        is_melee_weapon = (
            "fist" in current_weapon
            or "chainsaw" in current_weapon
            or "ripter" in current_weapon
            or "melee" in current_weapon
        )

        if action == "shoot" and not config.get("allow_shoot", False):
            return "move_forward"

        if action == "melee_attack" and not config.get("allow_melee", False):
            return "move_forward"

        if (
            action == "shoot"
            and config.get("require_enemy_visible_to_shoot", False)
            and not enemy_visible
        ):
            return "move_forward"

        if action == "shoot" and ammo <= 0:
            if is_melee_weapon and enemy_visible:
                return "melee_attack"
            return "swap_weapon"

        # New: prevent early/random weapon swapping.
        if action == "swap_weapon":
            if ammo > 5 and not enemy_visible:
                return "move_forward"

            if ammo > 5 and enemy_visible:
                return "shoot"

        if action == "melee_attack" and not enemy_visible:
            return "move_forward"

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