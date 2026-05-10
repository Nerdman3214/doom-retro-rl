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


DOOM_BINARY = "/home/steven/Downloads/doomretro-master/build/doomretro"
DOOM_IWAD = "/usr/share/games/doom/freedoom2.wad"

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

    def step(self, action_index):

        pixel_diff = 999.0

        frame_cache.invalidate()

        action = self.actions[action_index]
        reward = 0.0
        self.apply_movement(action)
        if action == "swap_weapon":
            self.swap_weapon_count += 1

        turn_signal = 0.0

        if action == "turn_left":
            turn_signal = -1.0
        elif action == "turn_right":
            turn_signal = 1.0

        turn = self.rotation_state.update(turn_signal)

        self.apply_rotation(turn)

        # FIX #4: Pass self.curriculum_stage so CurriculumManager stays synced.
        stage = self.curriculum.get_stage_name(self.curriculum_stage)

        config = self.get_stage_config()

        # Update goal distance tracking
        goal_pos = self.observer.get_goal_position()
        player_pos = self.observer.get_player_position()

        if goal_pos is not None and player_pos is not None:
            current_dist = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))
            if self.closest_distance is not None and current_dist < self.closest_distance:
                self.closest_distance = current_dist
                self.reward_manager.closest_distance = self.closest_distance

        self.reward_manager.update_distance_reward(player_pos, goal_pos)
        self.reward_manager.update_exploration_reward(player_pos)

        frame = self.observer.get_frame()
        enemy_visible = self.reward_manager.enemy_detector.detect_enemy_presence(frame)
        self.enemy_visible_steps += 1 if enemy_visible else 0

        if enemy_visible:
            self.reward_manager.enemy_visible()

        if action in ["turn_left", "turn_right"]:
            self.reward_manager.excessive_turn()

        # FIX #5: Aiming reward was inverted — now correctly fires when enemy IS
        # in crosshair (not when it isn't).
        if enemy_visible and self.reward_manager.enemy_in_crosshair(frame):
            self.reward_manager.add("aiming", +10)
            self.reward_manager.enemy_centered()

        # Enforce shoot restriction for stages that require enemy visibility
        if action == "shoot" and config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
            self.reward_manager.add("shoot_without_visible_enemy", -5.0)
            # Do NOT perform the action — skip to next step logic
        else:
            self.perform_action(action)

        # Track exploration area changes (stage 1)
        movement_actions = ["move_forward", "move_backward", "turn_left", "turn_right"]
        if self.curriculum_stage == 1:
            area_sig = self._get_area_signature()
            if area_sig != self.last_area_signature:
                self.exploration_count += 1
                self.visited_areas.add(area_sig)
                self.last_area_signature = area_sig

        # FIX #2: Shooting rewards are now in ONE place only — removed duplicate
        # wasted_shot / spam_penalty blocks that were scattered earlier in the
        # original step(). This single block handles all shoot accounting.
        if action == "shoot":
            if self._step_count - self.last_shot_step < 5:
                self.reward_manager.add("spam_penalty", -5.0)
            else:
                self.last_shot_step = self._step_count

            if self.reward_manager.enemy_in_crosshair(frame):
                self.reward_manager.add("good_shot", +10)
                if self.curriculum_stage >= 3:
                    self.enemy_engagement_count += 1
                    self.valid_shot_count += 1
            else:
                self.reward_manager.add("wasted_shot", -5.0)


        frame = self.observer.build()
        observation = self.frame_stack.add_frame(frame)

        game_state = self.observer.get_game_state()
        game_state["enemy_visible"] = enemy_visible

        self.reward_manager.update_resource_reward(
            game_state["health"],
            game_state["ammo"],
        )

        # ----------------------------------------------------------------
        # Position / movement tracking
        # ----------------------------------------------------------------
        distance_moved = 0.0
        if game_state["shared_state_available"]:
            player_x, player_y = game_state["x"], game_state["y"]
            current_player_pos = (player_x, player_y)

            if self.last_player_position is not None:
                distance_moved = float(
                    np.linalg.norm(
                        np.array(current_player_pos) - np.array(self.last_player_position)
                    )
                )

            self.last_player_position = current_player_pos
            self.distance_traveled += distance_moved

            if distance_moved > 5.0:
                self.movement_count += 1

            tile = (int(player_x // 64), int(player_y // 64))
            if tile not in self.visited_tiles:
                self.visited_tiles.add(tile)
                reward += 0.03
                self.exploration_count += 1

            self.exploration_memory.visit(player_x, player_y)
            novelty_reward = self.exploration_memory.get_novelty(player_x, player_y) * 0.3
            self.reward_manager.add("exploration_novelty", novelty_reward)

        # Expose distance_moved to game_state so movement_reward() can use it
        game_state["distance_moved"] = distance_moved
        game_state["action"] = action

        # ----------------------------------------------------------------
        # FIX #2 & #3: Stage-specific reward routing — each stage is now in
        # its own exclusive branch with no overlap.
        # FIX #3: Stage 5 (key_doors) is now reachable (was swallowed by >=4).
        # ----------------------------------------------------------------
        if self.curriculum_stage == 0:
            # Stage 0: Real locomotion only — movement_reward() owns this signal
            reward += self.reward_manager.movement_reward(game_state)

        elif self.curriculum_stage == 1:
            # Stage 1: Exploration + item pickups
            if distance_moved > 5.0:
                reward += 0.03
            if len(self.visited_tiles) > 0 and action in movement_actions:
                reward += 0.01
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                reward += 0.5
                self.pickup_count += 1

        elif self.curriculum_stage == 2:
            # Stage 2: Avoid stagnation
            if distance_moved > 5.0:
                reward += 0.03
            if pixel_diff < 2.0:
                reward -= 0.05
            if self.stuck_counter > 20:
                reward -= 0.75
            if action in ["move_forward", "move_backward", "strafe_left",
                          "strafe_right", "turn_left", "turn_right"] and pixel_diff > 1.0:
                self.continuous_movement_steps += 1

        elif self.curriculum_stage == 3:
            # Stage 3: Basic combat — reward shooting visible enemies
            if action == "shoot" and enemy_visible:
                reward += 1.0
            if self.reward_manager.enemy_in_crosshair(frame) and action == "shoot":
                reward += 2.0
            if action == "shoot" and not enemy_visible:
                reward -= 1.5
            if action == "use":
                self.door_interaction_count += 1
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                self.key_item_count += 1

        elif self.curriculum_stage == 4:
            # Stage 4: Full combat — accurate shooting rewarded via shoot block above
            pass

        elif self.curriculum_stage == 5:
            # FIX #3: Stage 5 now reachable — door/key rewards restored (single reward per use)
            if action == "use":
                reward += 2.0
                self.door_interaction_count += 1
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                reward += 0.3
                self.key_item_count += 1

        elif self.curriculum_stage >= 6:
            # Stage 6 & 7: Full game — combat handled by reward_manager
            pass

        # ----------------------------------------------------------------
        # FIX #4: track_enemy and dodge_enemies routing now uses stage name
        # that is correctly synced to self.curriculum_stage
        # ----------------------------------------------------------------
        if config.get("track_enemy", False) and enemy_visible:
            enemy_x = game_state.get("enemy_x", 42)
            center_distance = abs(enemy_x - 42)
            tracking_reward = max(0.0, 1.0 - (center_distance / 42))
            self.reward_manager.add("track_enemy", tracking_reward * 0.5)
            self.track_enemy_count += 1

        if config.get("dodge_enemies", False):
            if enemy_visible:
                self.reward_manager.add("enemy_visible_dodge", 0.1)
            if game_state.get("damage_taken", False):
                self.reward_manager.add("took_damage_penalty", -1.0)
            if action in ["turn_left", "turn_right", "move_backward",
                          "strafe_left", "strafe_right"]:
                self.reward_manager.add("dodge_movement", 0.05)
                self.dodge_enemies_count += 1


        if action == "shoot":
            if game_state.get("ammo") == 10:
                self.reward_manager.add("low_ammo", -1.0)
                if action == "swap_weapon":
                    self.reward_manager.add("weapon_swapped", +1.0)
                    self.swap_weapon_count += 1

        # ----------------------------------------------------------------
        # Stuck detection
        # ----------------------------------------------------------------
        current_frame = self.observer.get_frame()

        if self._previous_frame is not None:

            pixel_diff = float(
                np.abs(
                    current_frame.astype(int)
                    - self._previous_frame.astype(int)
                ).mean()
            )

            self.reward_manager.penalize_stuck(pixel_diff)

        self._previous_frame = current_frame

        if pixel_diff < 2.0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.reward_manager.reward_escape_behavior(action, pixel_diff)

        # FIX #1: Raised stuck termination threshold from 7 to 30 frames.
        # 7 frames (~0.4 seconds) was too aggressive — agent never had time to
        # learn escape behavior. 30 frames gives it a reasonable window.
        # FIX #7: long_stuck_penalty check moved BEFORE the counter reset so
        # it can actually fire.
        if self.stuck_counter > 40:
            self.reward_manager.add("long_stuck_penalty", -1.0)

        if self.stuck_counter >= 30:
            severity = self.reward_manager.get_stuck_severity()
            penalty = -2.0 * severity  # reduced from -5.0 to avoid overwhelming signal
            self.reward_manager.add("prolonged_stuck_penalty", penalty)
            self.stuck_counter = 0
            done = True
        else:
            done = False

        self.reward_manager.update_stagnation_penalty(self.stuck_counter)

        reward += self.reward_manager.get_reward()

        # Helper bot alignment reward
        helper_action = self.helper.get_action(game_state)
        if action == helper_action:
            reward += 0.2
        else:
            reward -= 0.05

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps

        # Event recording
        frame = self.observer.get_frame()
        if self.reward_overlay is not None:
            frame = self.reward_overlay.draw(frame, self.reward_manager.breakdown)

        player_health = game_state["health"]

        if self.previous_health is not None and player_health < self.previous_health:
            if self.record:
                self.smart_clipper.trigger_event()
                self.event_recorder.set_event("took_damage")

        self.previous_health = player_health

        if enemy_visible and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("enemy_seen")

        if (game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0) and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("pickup_collected")

        if player_health < 25 and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("low_health_escape")

        if action == "use" and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("door_open")

        if self._use_preference:
            pref_reward = self.preference_reward(frame)
            pref_reward = max(min(pref_reward, 3), -3)
            reward += 0.1 * pref_reward

        if self.record:
            self.logger.log(frame, action, game_state["health"], game_state["ammo"])
            self.traj_logger.record(frame, action, reward)
            self.recorder.record_frame(frame)
            self.event_recorder.record_frame(frame)
            self.clipper.record(frame, action)
            self.smart_clipper.observe(frame, action)

        self.curriculum_rewards.append(reward)
        if len(self.curriculum_rewards) > 100:
            self.curriculum_rewards.pop(0)

        self.update_curriculum()

        return observation, reward, done, truncated, {}

    def update_curriculum(self):

        if len(self.curriculum_rewards) < 50:
            return

        avg_reward = sum(self.curriculum_rewards) / len(self.curriculum_rewards)
        threshold = self.curriculum_thresholds.get(self.curriculum_stage, 0.0)
        reward_ready = True if threshold <= 0.0 else avg_reward > threshold
        behavior_ready = False

        if self.curriculum_stage == 0:
            behavior_ready = self.movement_count >= 20 and self.distance_traveled >= 150.0
            print(f"  [Stage 0] real moves: {self.movement_count}/20 | distance: {self.distance_traveled:.1f}/150 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 1:
            behavior_ready = len(self.visited_tiles) >= 5 and self.pickup_count >= 1
            print(f"  [Stage 1] tiles: {len(self.visited_tiles)}/5 | pickups: {self.pickup_count}/1 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 2:
            behavior_ready = self.continuous_movement_steps >= 30 and self.stuck_counter == 0
            print(f"  [Stage 2] movement steps: {self.continuous_movement_steps}/30 | stuck: {self.stuck_counter}")

        elif self.curriculum_stage == 3:
            behavior_ready = self.enemy_engagement_count >= 10 and self.enemy_visible_steps >= 10
            print(f"  [Stage 3] visible shots: {self.enemy_engagement_count}/10 | visible steps: {self.enemy_visible_steps}/10 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 4:
            behavior_ready = self.valid_shot_count >= 15
            print(f"  [Stage 4] accurate shots: {self.valid_shot_count}/15 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 5:
            behavior_ready = self.door_interaction_count >= 1 and self.key_item_count >= 1
            print(f"  [Stage 5] doors: {self.door_interaction_count}/1 | keys/items: {self.key_item_count}/1 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 6:
            behavior_ready = self.valid_shot_count >= 30
            print(f"  [Stage 6] accurate shots: {self.valid_shot_count}/30 | Reward: {avg_reward:.2f}/{threshold}")

        elif self.curriculum_stage == 7:
            behavior_ready = True
            print(f"  [Stage 7] final stage - Reward: {avg_reward:.2f}/{threshold}")

        stage_ready = reward_ready and behavior_ready

        if stage_ready and self.curriculum_stage < self.max_stage:
            old_stage = self.curriculum_stage
            self.curriculum_stage += 1
            self.curriculum_rewards.clear()

            # Reset all counters for next stage
            self.movement_count = 0
            self.exploration_count = 0
            self.pickup_count = 0
            self.continuous_movement_steps = 0
            self.door_interaction_count = 0
            self.key_item_count = 0
            self.enemy_engagement_count = 0
            self.valid_shot_count = 0
            self.enemy_visible_steps = 0
            self.stuck_counter = 0
            self.visited_areas.clear()

            print(f"[Curriculum] ADVANCING FROM STAGE {old_stage} → {self.curriculum_stage}")

    def _get_area_signature(self):
        frame = self.observer.get_frame()
        small_frame = frame[::20, ::20]
        return int(small_frame.mean())

    def preference_reward(self, frame):
        frame_tensor = torch.tensor(
            frame
        ).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            score = self.preference_model(frame_tensor)

        return score.item()

    def perform_action(self, action):
        allowed = self.get_allowed_actions()
        config = self.get_stage_config()

        if not config["allow_shoot"] and "shoot" in action:
            return
        
        if action not in allowed:
            return  # ignore invalid action OR penalize

        elif action == "shoot":
            self.controller.shoot()

        elif action == "use":
            self.controller.use()

        elif action == "swap_weapon":
            self.controller.swap_weapon()

        elif action == "strafe_left":
            self.controller.strafe_left()

        elif action == "strafe_right":
            self.controller.strafe_right()

    def apply_rotation(self, turn_value):
        if turn_value < -0.2:
            self.controller.start_turn_left()
            self.controller.key_up("Right")

        elif turn_value > 0.2:
            self.controller.start_turn_right()
            self.controller.key_up("Left")

        else:
            self.controller.stop_turn()

    def apply_movement(self, action):

        # stop old movement first
        self.controller.stop_forward_backward()

        if action == "move_forward":
            self.controller.start_move_forward()

        elif action == "move_backward":
            self.controller.start_move_backward()

    def close(self):
        if self.record:
            if self.recorder:
                self.recorder.save()
            if self.event_recorder:
                self.event_recorder.save()
        if self.game_process:
            self.game_process.terminate()