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


DOOM_BINARY = "/home/steven/Downloads/doomretro-master/build/doomretro"
DOOM_IWAD = "/usr/share/games/doom/freedoom1.wad"


class DoomEnv(gym.Env):

    def __init__(self, launch_doom=False, record=True):

        super(DoomEnv, self).__init__()

        # Only launch DOOM automatically during RL training.
        # For inference (play_human.py), start DOOM manually first.
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
        self.curriculum_rewards = []
        self.curriculum_thresholds = {
            0: 5000,
            1: 10000,
            2: 15000,
            3: 25000,
            4: 35000,
            5: 45000,
            6: 55000,
            7: 65000
        }
        self.max_stage = 7
        self.enemy_visible_steps = 0
        self.initial_distance = None
        self.closest_distance = None
        
        # Stage-specific behavior counters
        self.movement_count = 0      # Stage 0: count movements
        self.exploration_count = 0            # Stage 1: count new areas and item pickups
        self.pickup_count = 0                 # Stage 1: count important item pickups
        self.continuous_movement_steps = 0   # Stage 2: count non-stuck steps
        self.door_interaction_count = 0       # Stage 3: count door interactions
        self.key_item_count = 0               # Stage 3: count important item pickups/keys
        self.enemy_engagement_count = 0       # Stage 4: count shots at visible enemies
        self.valid_shot_count = 0             # Stage 5+: count accurate shots
        self.level_completion_count = 0       # Stage 7: track level completion
        self.visited_areas = set()            # Track unique areas for exploration

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
            shape=(9, 84, 84),  # channels-first stack for SB3 CNN
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
            # Start keyboard listener once here, not every step
            self._kb_listener = keyboard.Listener(
                on_press=self.recorder.on_press,
                on_release=self.recorder.on_release,
            )
            self._kb_listener.start()
        else:
            self._kb_listener = None
        
        # Initialize area tracking
        self.last_area_signature = None

    def get_stage_config(self):

        return {
            0: {
                "name": "movement",
                "allow_shoot": False,
                "enemy_enabled": False,
                "goal": "move"
            },
            1: {
                "name": "game_basics",
                "allow_shoot": False,
                "enemy_enabled": False,
                "goal": "explore_and_pick_items"
            },
            2: {
                "name": "avoid_getting_stuck",
                "allow_shoot": False,
                "enemy_enabled": False,
                "goal": "avoid_stuck"
            },
            3: {
                "name": "combat_basic",
                "visible": True,
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "max_enemies": 1,
                "goal": "defeat_enemies"
            },
            4: {
                "name": "combat_full",
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "max_enemies": min(10, self._step_count // 200),
                "goal": "defeat_enemies"
            },
            5: {
                "name": "key_doors",
                "allow_shoot": False,
                "enemy_enabled": False,
                "goal": "open_doors"
            },
            6: {
                "name": "full_game",
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "max_enemies": 1000,
                "goal": "defeat_enemies"
            },
            7: {
                "name": 'complete_level',
                "allow_shoot": True,
                "require_enemy_visible_to_shoot": True,
                "enemy_enabled": True,
                "max_enemies": 5000,
                "goal": "finish_level"
            }
        }[self.curriculum_stage]

    def reset(self, seed=None, options=None):
        self._step_count = 0
        
        # Reset stage-specific counters on episode reset
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

        goal_pos = self.observer.get_goal_position()
        player_pos = self.observer.get_player_position()

        dist = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

        self.initial_distance = dist
        self.closest_distance = dist
                

        self.visited_areas.clear()
        
        # Reset reward manager state for new episode
        self.reward_manager.reset()
        self.reward_manager.initial_distance = dist
        self.reward_manager.closest_distance = dist

        config = self.get_stage_config()

        print(f"[Curriculum] Stage {self.curriculum_stage}: {config['name']}")

        # Press Enter to dismiss any death/game-over screen and respawn
        import subprocess
        wid = self.controller.window_id
        if wid:
            subprocess.call(
                ["xdotool", "windowfocus", "--sync", wid, "key", "Return"],
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.5)  # wait for respawn animation

        frame_cache.invalidate()

        self.previous_health = None
        self._previous_frame = None

        frame = self.observer.build()
        observation = self.frame_stack.reset(frame)
        return observation, {}


    def step(self, action_index):
        # Invalidate the shared frame cache so all detectors share one fresh capture
        frame_cache.invalidate()

        self.update_curriculum()

        action = self.actions[action_index]

        # Update closest distance to goal for normalized reward
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

        config = self.get_stage_config()

        if enemy_visible and not self.reward_manager.enemy_in_crosshair(frame):
            self.reward_manager.add("aiming", +0.05)

        if action == "shoot" and config.get("require_enemy_visible_to_shoot", False) and not enemy_visible:
            self.reward_manager.add("shoot_without_visible_enemy", -1.0)
        else:
            self.perform_action(action)

        # Track stage-specific behavior
        movement_actions = ["move_forward", "move_backward", "turn_left", "turn_right"]
        if self.curriculum_stage == 0 and action in movement_actions:
            self.movement_count += 1
        
        if self.curriculum_stage == 1:
            area_sig = self._get_area_signature()
            if area_sig != self.last_area_signature:
                self.exploration_count += 1
                self.visited_areas.add(area_sig)
                self.last_area_signature = area_sig

        if self.curriculum_stage >= 4 and action == "shoot":
            if enemy_visible:
                self.enemy_engagement_count += 1
            if self.reward_manager.enemy_in_crosshair(frame):
                self.valid_shot_count += 1

        if action == "shoot":
            if self._step_count - self.last_shot_step < 5:
                self.reward_manager.add("spam_penalty", -0.3)
            else:
                self.last_shot_step = self._step_count

            if not self.reward_manager.enemy_in_crosshair(frame):
                self.reward_manager.add("wasted_shot", -0.2)
            else:
                self.reward_manager.add("good_shot", +0.3)

        frame = self.observer.build()
        observation = self.frame_stack.add_frame(frame)

        game_state = self.observer.get_game_state()
        game_state["enemy_visible"] = enemy_visible
        self.reward_manager.update_resource_reward(
            game_state["health"],
            game_state["ammo"],
        )

        if game_state["shared_state_available"]:
            player_x, player_y = game_state["x"], game_state["y"]
            self.exploration_memory.visit(player_x, player_y)
            novelty_reward = self.exploration_memory.get_novelty(player_x, player_y) * 0.3
            self.reward_manager.add("exploration_novelty", novelty_reward)
            if novelty_reward > 0.2:
                self.exploration_count += 1

        # Stuck detection: compare current frame to previous
        current_frame = self.observer.get_frame()
        pixel_diff = 0
        if self._previous_frame is not None:
            pixel_diff = float(
                np.abs(current_frame.astype(int) - self._previous_frame.astype(int)).mean()
            )
            self.reward_manager.penalize_stuck(pixel_diff)
        self._previous_frame = current_frame

        reward = self.reward_manager.get_reward()

        # Update progression counters based on recently detected game events
        if self.curriculum_stage == 1:
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                self.pickup_count += 1

        if self.curriculum_stage == 2 and action in ["move_forward", "move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right"] and pixel_diff > 1.0:
            self.continuous_movement_steps += 1

        if self.curriculum_stage == 3:
            if action == "use":
                self.door_interaction_count += 1
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                self.key_item_count += 1

        config = self.get_stage_config()

        # Stage-specific reward shaping
        if self.curriculum_stage == 0:
            # Stage 0: Focus on any movement and exploration
            if action in movement_actions:
                reward += 0.05
            else:
                reward -= 0.01

        elif self.curriculum_stage == 1:
            # Stage 1: Reward any movement, exploration, and item pickups
            reward += 0.02
            if action in ["move_forward", "move_backward"]:
                reward += 0.03
            if action in ["turn_left", "turn_right"]:
                reward += 0.01
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                reward += 0.5

        elif self.curriculum_stage == 2:
            # Stage 2: Reward all movement, penalize staying still
            if action in ["move_forward", "move_backward", "turn_left", "turn_right"]:
                reward += 0.05
            elif action == "use":
                reward += 0.04
            else:
                reward -= 0.02
            
            if pixel_diff < 1.5:
                reward -= 0.02


        elif self.curriculum_stage == 3:
            # Stage 4: Reward shooting visible enemies
            if action == "shoot" and enemy_visible:
                reward += 0.10
            if self.reward_manager.enemy_in_crosshair(frame) and action == "shoot":
                reward += 0.20

        elif self.curriculum_stage >= 4:
            # Stage 5+: Combat mastery - rewards handled by reward_manager
            pass

        elif self.curriculum_stage == 5:
            # Stage 5: Reward door interaction and item/key awareness
            if action == "use":
                reward += 0.15
            if action == "use":
                reward += 2.0
            if game_state["health_delta"] > 0 or game_state["ammo_delta"] > 0:
                reward += 0.3
            reward += 0.01

        helper_action = self.helper.get_action(game_state)
        if action == helper_action:
            reward += 0.2
        else:
            reward -= 0.05

        if pixel_diff < 1.5:  # Threshold for being "stuck" (more sensitive)
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.reward_manager.reward_escape_behavior(action, pixel_diff)
        
        if self.stuck_counter >= 7:  # Terminate stuck episode earlier
            severity = self.reward_manager.get_stuck_severity()
            penalty = -5.0 * severity
            self.reward_manager.add("prolonged_stuck_penalty", penalty)
            self.stuck_counter = 0
            done = True
        else:
            done = False

        self.reward_manager.update_stagnation_penalty(self.stuck_counter)
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps

        frame = self.observer.get_frame()
        if self.reward_overlay is not None:
            frame = self.reward_overlay.draw(frame, self.reward_manager.breakdown)

        player_health = game_state["health"]

        if self.previous_health is not None:
            if player_health < self.previous_health:
                if self.record:
                    self.smart_clipper.trigger_event()
                    self.event_recorder.set_event("took_damage")

        self.previous_health = player_health

        # Enemy detection event trigger
        enemy_detected = enemy_visible
        if enemy_detected and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("enemy_seen")

        # Resource pickup trigger
        if (game_state["health_delta"] > 0 or
                game_state["ammo_delta"] > 0) and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("pickup_collected")

        # Low-health survival trigger
        if player_health < 25 and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("low_health_escape")

        # Door interaction trigger
        if action == "use" and self.record:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("door_open")

        if self._use_preference:
            pref_reward = self.preference_reward(frame)
            pref_reward = max(min(pref_reward, 3), -3)
            reward += 0.1 * pref_reward

        if self.record:
            self.logger.log(frame, action, game_state["health"],
                            game_state["ammo"])

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
        threshold = self.curriculum_thresholds.get(self.curriculum_stage, 1e9)
        reward_ready = avg_reward > threshold
        behavior_ready = False

        if self.curriculum_stage == 0:
            # Stage 0: Require 100+ forward movements
            behavior_ready = self.movement_count >= 50
            print(f"  [Stage 0] moves: {self.movement_count}/50 | Reward: {avg_reward:.1f}/{threshold}")
        
        elif self.curriculum_stage == 1:
            # Stage 1: Require exploration and basic resource pickup
            behavior_ready = len(self.visited_areas) >= 1 and self.pickup_count >= 3
            print(f"  [Stage 1] Areas explored: {len(self.visited_areas)}/1 | Pickups: {self.pickup_count}/3 | Reward: {avg_reward:.1f}/{threshold}")
        
        elif self.curriculum_stage == 2:
            # Stage 2: Require 80+ continuous movement steps (not stuck)
            behavior_ready = self.continuous_movement_steps >= 20 and self.stuck_counter == 0
            print(f"  [Stage 2] Movement steps: {self.continuous_movement_steps}/20 | Stuck counter: {self.stuck_counter}")
        
        elif self.curriculum_stage == 3:
            # Stage 4: Require 20+ shots at visible enemies + 10+ steps seeing enemies
            behavior_ready = self.enemy_engagement_count >= 20 and self.enemy_visible_steps >= 10
            print(f"  [Stage 4] Enemy shots: {self.enemy_engagement_count}/20 | Visible: {self.enemy_visible_steps}/10")
        
        elif self.curriculum_stage == 4:
            # Stage 5: Require 40+ accurate shots
            behavior_ready = self.valid_shot_count >= 40
            print(f"  [Stage 5] Accurate shots: {self.valid_shot_count}/40 | Reward: {avg_reward:.1f}/{threshold}")

        elif self.curriculum_stage == 5:
            # Stage 3: Require door usage plus item/key awareness
            behavior_ready = self.door_interaction_count >= 1 and self.key_item_count >= 1
            print(f"  [Stage 3] Doors opened: {self.door_interaction_count}/1 | Items collected: {self.key_item_count}/1 | Reward: {avg_reward:.1f}/{threshold}")
        
        elif self.curriculum_stage == 6:
            # Stage 6: Require 60+ accurate shots (sustained combat)
            behavior_ready = self.valid_shot_count >= 60
            print(f"  [Stage 6] Accurate shots: {self.valid_shot_count}/60 | Reward: {avg_reward:.1f}/{threshold}")
        
        elif self.curriculum_stage == 7:
            # Stage 7: Just maintain good reward
            behavior_ready = True
            print(f"  [Stage 7] Final stage - Reward: {avg_reward:.1f}/{threshold}")

        stage_ready = reward_ready and behavior_ready

        if stage_ready:
            if self.curriculum_stage < self.max_stage:
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
        """Get a simple signature of current screen state for exploration tracking."""
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

        config = self.get_stage_config()

        # Disable shooting in early stages
        if not config["allow_shoot"] and "shoot" in action:
            return

        # Normal execution
        if action == "move_forward":
            self.controller.move_forward()

        elif action == "move_backward":
            self.controller.move_backward()

        elif action == "move_left":
            self.controller.move_left()

        elif action == "move_right":
            self.controller.move_right()

        elif action == "turn_left":
            self.controller.turn_left()

        elif action == "turn_right":
            self.controller.turn_right()

        elif action == "shoot":
            self.controller.shoot()

        elif action == "move_backward":
            self.controller.move_backward()

        elif action == "use":
            self.controller.use()

        elif action == "turn_left_shoot":
            self.controller.turn_left_shoot()

        elif action == "turn_right_shoot":
            self.controller.turn_right_shoot()

        elif action == "move_forward_shoot":
            self.controller.move_forward_shoot()

        elif action == "move_backward_shoot":
            self.controller.move_backward_shoot()

        elif action == "swap_weapon":
            self.controller.swap_weapon()

        return

    def close(self):
        if self.record:
            if self.recorder:
                self.recorder.save()
            if self.event_recorder:
                self.event_recorder.save()
        if self.game_process:
            self.game_process.terminate()
