import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from observation.native_frame_packer import NativeFramePacker

try:
    import vizdoom as vzd
except Exception as e:
    vzd = None
    _VIZDOOM_IMPORT_ERROR = e
else:
    _VIZDOOM_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IWAD = "/usr/share/games/doom/freedoom1.wad"


class VizDoomEnv(gym.Env):
    """
    Optional ViZDoom backend for faster RL experiments and geometry supervision.

    This environment is separate from env/doom_env.py.

    Observation:
        Dict with:
        - screen: RGB gameplay frame, 3x84x84
        - depth: normalized depth buffer, 1x84x84
        - labels: labels buffer, 1x84x84
        - automap: automap buffer, 3x84x84

    It is useful for:
        - training navigation policies with real depth
        - collecting geometry labels
        - debugging wall/obstacle perception

    It should not replace Doom Retro immediately.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    ACTIONS = [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "shoot",
        "use",
    ]

    def __init__(
        self,
        iwad_path=DEFAULT_IWAD,
        scenario_path=None,
        visible=False,
        frame_skip=4,
        max_episode_steps=1000,
        use_automap=True,
        use_labels=True,
        use_depth=True,
        screen_size=(84, 84),
        observation_mode="compact",
    ):
        super().__init__()

        if vzd is None:
            raise ImportError(
                f"ViZDoom could not be imported: {_VIZDOOM_IMPORT_ERROR}"
            )

        self.iwad_path = iwad_path
        self.scenario_path = scenario_path
        self.visible = visible
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.use_automap = use_automap
        self.use_labels = use_labels
        self.use_depth = use_depth
        self.screen_size = screen_size
        self.observation_mode = observation_mode
        self.native_packer = NativeFramePacker(out_size=screen_size)

        self.game = None
        self.step_count = 0
        self.last_health = None
        self.last_ammo = None
        self.last_kill_count = 0
        self.last_item_count = 0

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        if self.observation_mode == "compact":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(4, screen_size[1], screen_size[0]),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, screen_size[1], screen_size[0]),
                        dtype=np.uint8,
                    ),
                    "depth": spaces.Box(
                        low=0,
                        high=255,
                        shape=(1, screen_size[1], screen_size[0]),
                        dtype=np.uint8,
                    ),
                    "labels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(1, screen_size[1], screen_size[0]),
                        dtype=np.uint8,
                    ),
                    "automap": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, screen_size[1], screen_size[0]),
                        dtype=np.uint8,
                    ),
                }
            )

        self._init_game()

    def _init_game(self):
        game = vzd.DoomGame()

        if self.scenario_path:
            game.load_config(str(self.scenario_path))

        if self.iwad_path and os.path.exists(self.iwad_path):
            game.set_doom_game_path(self.iwad_path)

        game.set_window_visible(self.visible)
        game.set_mode(vzd.Mode.PLAYER)

        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_screen_format(vzd.ScreenFormat.RGB24)

        game.set_depth_buffer_enabled(self.use_depth)
        game.set_labels_buffer_enabled(self.use_labels)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_automap_buffer_enabled(self.use_automap)

        # Make automap more useful as a geometry signal.
        if self.use_automap:
            game.set_automap_mode(vzd.AutomapMode.OBJECTS)
            game.set_automap_rotate(False)
            game.set_automap_render_textures(False)

        game.set_available_buttons(
            [
                vzd.Button.MOVE_FORWARD,
                vzd.Button.MOVE_BACKWARD,
                vzd.Button.TURN_LEFT,
                vzd.Button.TURN_RIGHT,
                vzd.Button.MOVE_LEFT,
                vzd.Button.MOVE_RIGHT,
                vzd.Button.ATTACK,
                vzd.Button.USE,
            ]
        )

        game.set_available_game_variables(
            [
                vzd.GameVariable.HEALTH,
                vzd.GameVariable.ARMOR,
                vzd.GameVariable.AMMO2,
                vzd.GameVariable.SELECTED_WEAPON,
                vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.ITEMCOUNT,
                vzd.GameVariable.POSITION_X,
                vzd.GameVariable.POSITION_Y,
                vzd.GameVariable.POSITION_Z,
                vzd.GameVariable.ANGLE,
            ]
        )

        game.init()
        self.game = game

    def _action_to_buttons(self, action_index):
        action_name = self.ACTIONS[int(action_index)]

        buttons = [0] * len(self.ACTIONS)

        if action_name == "move_forward":
            buttons[0] = 1
        elif action_name == "move_backward":
            buttons[1] = 1
        elif action_name == "turn_left":
            buttons[2] = 1
        elif action_name == "turn_right":
            buttons[3] = 1
        elif action_name == "strafe_left":
            buttons[4] = 1
        elif action_name == "strafe_right":
            buttons[5] = 1
        elif action_name == "shoot":
            buttons[6] = 1
        elif action_name == "use":
            buttons[7] = 1

        return buttons, action_name

    def _resize_chw(self, img, channels=3):
        w, h = self.screen_size

        if img is None:
            if channels == 1:
                return np.zeros((1, h, w), dtype=np.uint8)
            return np.zeros((3, h, w), dtype=np.uint8)

        arr = np.asarray(img)

        if arr.ndim == 2:
            resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
            return resized[None, :, :].astype(np.uint8)

        # ViZDoom RGB buffer is usually CHW or HWC depending on format/version.
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        return np.transpose(resized, (2, 0, 1)).astype(np.uint8)

    def _depth_to_uint8(self, depth):
        if depth is None:
            w, h = self.screen_size
            return np.zeros((1, h, w), dtype=np.uint8)

        arr = np.asarray(depth).astype(np.float32)

        # ViZDoom depth values can vary depending on version/config.
        # Normalize per-frame for learning/debugging.
        finite = np.isfinite(arr)

        if not finite.any():
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            valid = arr[finite]
            lo = float(valid.min())
            hi = float(valid.max())

            if hi <= lo:
                norm = np.zeros_like(arr, dtype=np.uint8)
            else:
                norm = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

        return self._resize_chw(norm, channels=1)

    def _labels_to_uint8(self, labels):
        if labels is None:
            w, h = self.screen_size
            return np.zeros((1, h, w), dtype=np.uint8)

        arr = np.asarray(labels)

        if arr.ndim == 3:
            arr = arr[0]

        arr = arr.astype(np.uint8)
        return self._resize_chw(arr, channels=1)

    def _get_game_vars(self, state):
        values = {}

        if state is None or state.game_variables is None:
            return values

        names = [
            "health",
            "armor",
            "ammo",
            "selected_weapon",
            "kill_count",
            "item_count",
            "x",
            "y",
            "z",
            "angle",
        ]

        for name, value in zip(names, state.game_variables):
            values[name] = float(value)

        return values

    def _make_obs(self):
        state = self.game.get_state()

        if state is None:
            if self.observation_mode == "compact":
                h = self.screen_size[1]
                w = self.screen_size[0]
                return np.zeros((4, h, w), dtype=np.uint8)

            return {
                "screen": self._resize_chw(None, channels=3),
                "depth": self._resize_chw(None, channels=1),
                "labels": self._resize_chw(None, channels=1),
                "automap": self._resize_chw(None, channels=3),
            }

        screen = state.screen_buffer
        depth_raw = getattr(state, "depth_buffer", None)

        if self.observation_mode == "compact":
            if depth_raw is None:
                h = self.screen_size[1]
                w = self.screen_size[0]
                screen_chw = self.native_packer.pack_rgb(screen)
                depth_chw = np.zeros((1, h, w), dtype=np.uint8)
                return np.concatenate([screen_chw, depth_chw], axis=0)

            return self.native_packer.pack_compact_rgb_depth(screen, depth_raw)

        screen = self._resize_chw(screen, channels=3)
        depth = self._depth_to_uint8(depth_raw)
        labels = self._labels_to_uint8(getattr(state, "labels_buffer", None))
        automap = self._resize_chw(getattr(state, "automap_buffer", None), channels=3)

        return {
            "screen": screen,
            "depth": depth,
            "labels": labels,
            "automap": automap,
        }

    def _compute_reward(self, action_name, state):
        reward = 0.0

        vars_now = self._get_game_vars(state)

        health = vars_now.get("health")
        ammo = vars_now.get("ammo")
        kills = vars_now.get("kill_count", 0)
        items = vars_now.get("item_count", 0)

        # Small living cost to discourage doing nothing forever.
        reward -= 0.001

        if self.last_health is not None and health is not None:
            health_delta = health - self.last_health
            if health_delta < 0:
                reward += health_delta * 0.02

        if kills > self.last_kill_count:
            reward += 1.0 * (kills - self.last_kill_count)

        if items > self.last_item_count:
            reward += 0.25 * (items - self.last_item_count)

        if action_name == "move_forward":
            reward += 0.002

        self.last_health = health
        self.last_ammo = ammo
        self.last_kill_count = kills
        self.last_item_count = items

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.game.set_seed(int(seed))

        self.game.new_episode()
        self.step_count = 0

        state = self.game.get_state()
        vars_now = self._get_game_vars(state)

        self.last_health = vars_now.get("health")
        self.last_ammo = vars_now.get("ammo")
        self.last_kill_count = vars_now.get("kill_count", 0)
        self.last_item_count = vars_now.get("item_count", 0)

        obs = self._make_obs()

        info = {
            "backend": "vizdoom",
            "game_vars": vars_now,
        }

        return obs, info

    def step(self, action_index):
        buttons, action_name = self._action_to_buttons(action_index)

        self.game.make_action(buttons, self.frame_skip)
        self.step_count += 1

        done = self.game.is_episode_finished()
        truncated = self.step_count >= self.max_episode_steps

        state = None if done else self.game.get_state()
        reward = self._compute_reward(action_name, state)

        obs = self._make_obs()

        info = {
            "backend": "vizdoom",
            "action_name": action_name,
            "game_vars": self._get_game_vars(state),
        }

        if done:
            reward += float(self.game.get_total_reward())

        return obs, float(reward), bool(done), bool(truncated), info

    def render(self):
        state = self.game.get_state()

        if state is None:
            return None

        arr = np.asarray(state.screen_buffer)

        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        return arr

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
