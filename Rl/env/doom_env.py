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
import frame_cache
from Helper.helper_bot import HelperBot
from controller.doom_controller import DoomController
from observation.observation_builder import ObservationBuilder
from rewards.reward_manager import RewardManager
from action.action_space import ActionSpace
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

    def __init__(self, launch_doom=False):

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
        self.reward_manager = RewardManager()
        self.actions = ActionSpace().ACTIONS
        self.logger = BehaviorLogger()
        self.traj_logger = TrajectoryLogger()
        self.recorder = PlayerRecorder()
        self.event_recorder = EventRecorder()
        self.clipper = ClipGenerator(clip_length=90)
        self.preference_model = PreferenceModel()
        self.smart_clipper = SmartClipGenerator()
        self.helper = HelperBot()
        self.stuck_counter = 0 
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
            shape=(84, 84, 12),
            dtype=np.uint8
        )

        self.previous_health = None
        self._step_count = 0
        self._max_episode_steps = 1000
        self._previous_frame = None
        self.last_shot_step = -100

        # Start keyboard listener once here, not every step
        self._kb_listener = keyboard.Listener(
            on_press=self.recorder.on_press,
            on_release=self.recorder.on_release,
        )
        self._kb_listener.start()


    def reset(self, seed=None, options=None):
        self._step_count = 0

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
        observation = self.observer.build()
        return observation, {}


    def step(self, action_index):
        # Invalidate the shared frame cache so all detectors share one fresh capture
        frame_cache.invalidate()

        action = self.actions[action_index]

        self.perform_action(action)

        frame = self.observer.get_frame()
        enemy_visible = self.reward_manager.enemy_detector.detect_enemy_presence(frame)

        if enemy_visible and not self.reward_manager.enemy_in_crosshair(frame):
            self.reward_manager.add("aiming", +0.05)

        if action == "shoot":
            if self._step_count - self.last_shot_step < 5:
                self.reward_manager.add("spam_penalty", -0.3)
            else:
                self.last_shot_step = self._step_count

            if not self.reward_manager.enemy_in_crosshair(frame):
                self.reward_manager.add("wasted_shot", -0.2)
            else:
                self.reward_manager.add("good_shot", +0.3)

        observation = self.observer.build()

        game_state = self.observer.get_game_state()
        game_state["enemy_visible"] = enemy_visible

        # Stuck detection: compare current frame to previous
        current_frame = self.observer.get_frame()
        pixel_diff = 0
        if self._previous_frame is not None:
            pixel_diff = float(
                np.abs(current_frame.astype(int) - self._previous_frame.astype(int)).mean()
            )
            self.reward_manager.penalize_stuck(pixel_diff)
        self._previous_frame = current_frame

        reward = self.reward_manager.calculate_reward(action, game_state)

        helper_action = self.helper.get_action(game_state)
        if action == helper_action:
            reward += 0.2
        else:
            reward -= 0.05

        if pixel_diff < 2:  # Threshold for being "stuck"
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0


        if self.stuck_counter >= 10:  # If stuck for 10 consecutive steps
            self.stuck_counter = 0

        self._step_count += 1
        done = False
        truncated = self._step_count >= self._max_episode_steps

        frame = self.observer.get_frame()

        player_health = game_state["health"]

        if self.previous_health is not None:
            if player_health < self.previous_health:
                self.smart_clipper.trigger_event()
                self.event_recorder.set_event("took_damage")

        self.previous_health = player_health

        # Enemy detection event trigger
        enemy_detected = enemy_visible
        if enemy_detected:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("enemy_seen")

        # Resource pickup trigger
        if (self.reward_manager.breakdown.get("health_pickup", 0) > 0 or
                self.reward_manager.breakdown.get("ammo_pickup", 0) > 0):
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("pickup_collected")

        # Low-health survival trigger
        if player_health < 25:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("low_health_escape")

        # Door interaction trigger
        if self.reward_manager.breakdown.get("door_event", 0) > 0:
            self.smart_clipper.trigger_event()
            self.event_recorder.set_event("door_open")

        if self._use_preference:
            pref_reward = self.preference_reward(frame)
            pref_reward = max(min(pref_reward, 3), -3)
            reward += 0.1 * pref_reward

        self.logger.log(frame, action, game_state["health"],
                        game_state["ammo"])

        self.traj_logger.record(frame, action, reward)

        self.recorder.record_frame(frame)
        self.event_recorder.record_frame(frame)

        self.clipper.record(frame, action)

        self.smart_clipper.observe(frame, action)

        return observation, reward, done, truncated, {}

    def preference_reward(self, frame):

        frame_tensor = torch.tensor(
            frame
        ).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():

            score = self.preference_model(frame_tensor)

        return score.item()

    def perform_action(self, action):

        if action == "move_forward":
            self.controller.move_forward()

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

    def close(self):
        self.recorder.save()
        self.event_recorder.save()
        if self.game_process:
            self.game_process.terminate()
