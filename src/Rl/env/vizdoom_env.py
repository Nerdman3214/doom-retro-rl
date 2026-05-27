import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from observation.native_frame_packer import NativeFramePacker
from env.shared_doom_logic import SharedDoomLogic

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
        self.shared_logic = SharedDoomLogic()
        self.previous_game_state = None
        self.last_reward_debug = {}

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
        else:
            if self.iwad_path and os.path.exists(self.iwad_path):
                game.set_doom_game_path(self.iwad_path)
            else:
                raise FileNotFoundError(f"IWAD not found: {self.iwad_path}")

            # Important: force the game into an actual playable map.
            game.set_doom_map("E1M1")

        game.set_window_visible(self.visible)
        game.set_mode(vzd.Mode.PLAYER)

        # Headless/stability settings.
        if hasattr(game, "set_sound_enabled"):
            game.set_sound_enabled(False)

        if hasattr(game, "set_console_enabled"):
            game.set_console_enabled(False)

        # Start after map spawn settles.
        if hasattr(game, "set_episode_start_time"):
            game.set_episode_start_time(10)

        if hasattr(game, "set_episode_timeout"):
            game.set_episode_timeout(self.max_episode_steps * self.frame_skip)

        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_screen_format(vzd.ScreenFormat.RGB24)

        game.set_depth_buffer_enabled(self.use_depth)
        game.set_labels_buffer_enabled(self.use_labels)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_automap_buffer_enabled(self.use_automap)

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
                vzd.GameVariable.AMMO1,
                vzd.GameVariable.SELECTED_WEAPON,
                vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.ITEMCOUNT,
                vzd.GameVariable.POSITION_X,
                vzd.GameVariable.POSITION_Y,
                vzd.GameVariable.POSITION_Z,
                vzd.GameVariable.ANGLE,
            ]
        )

        print(
            "[vizdoom_env] init "
            f"iwad={self.iwad_path} "
            f"map=E1M1 "
            f"visible={self.visible} "
            f"depth={self.use_depth} "
            f"labels={self.use_labels} "
            f"automap={self.use_automap} "
            f"mode={self.observation_mode}"
        )

        game.init()
        self.game = game

    def _action_name_to_index(self, action_name):
        if action_name in self.ACTIONS:
            return self.ACTIONS.index(action_name)

        return 0

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
    
    def _state_to_game_state(self, state):
        """
        Convert ViZDoom GameState into the normalized shared game_state format.
        """

        values = self._get_game_vars(state)

        objects = []
        sectors = []

        if state is not None and getattr(state, "objects", None) is not None:
            for obj in state.objects:
                obj_data = {}

                for attr in [
                    "id",
                    "name",
                    "position_x",
                    "position_y",
                    "position_z",
                    "angle",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                    "width",
                    "height",
                ]:
                    if hasattr(obj, attr):
                        try:
                            value = getattr(obj, attr)
                            if isinstance(value, (int, float, str, bool)):
                                obj_data[attr] = value
                            else:
                                obj_data[attr] = float(value)
                        except Exception:
                            obj_data[attr] = str(getattr(obj, attr))

                objects.append(obj_data)

        if state is not None and getattr(state, "sectors", None) is not None:
            for sec in state.sectors:
                sec_data = {}

                for attr in [
                    "floor_height",
                    "ceiling_height",
                    "line_count",
                ]:
                    if hasattr(sec, attr):
                        try:
                            sec_data[attr] = float(getattr(sec, attr))
                        except Exception:
                            sec_data[attr] = str(getattr(sec, attr))

                sectors.append(sec_data)

        return {
            "health": values.get("health"),
            "armor": values.get("armor"),
            "ammo": values.get("ammo"),
            "selected_weapon": values.get("selected_weapon"),
            "kill_count": values.get("kill_count"),
            "item_count": values.get("item_count"),
            "x": values.get("x"),
            "y": values.get("y"),
            "z": values.get("z"),
            "angle": values.get("angle"),
            "objects": objects,
            "sectors": sectors,
        }

    def _make_obs(self):
        try:
            state = self.game.get_state()
        except Exception:
            state = None

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
        """
        Shared reward wrapper for ViZDoom.

        This converts ViZDoom state into our backend-independent game_state,
        extracts the depth buffer, and lets SharedDoomLogic compute reward.
        """

        current_game_state = self._state_to_game_state(state)

        depth_obs = None

        if state is not None:
            depth_raw = getattr(state, "depth_buffer", None)

            if depth_raw is not None:
                try:
                    depth_obs = self.native_packer.pack_depth(depth_raw)
                except Exception:
                    depth_obs = None

        reward, debug = self.shared_logic.compute_reward(
            previous_state=self.previous_game_state,
            current_state=current_game_state,
            action_name=action_name,
            depth_obs=depth_obs,
        )

        self.previous_game_state = current_game_state
        self.last_reward_debug = debug

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.game.set_seed(int(seed))

        self.game.new_episode()
        self.step_count = 0
        self.shared_logic.reset_episode()

        state = self.game.get_state()
        self.previous_game_state = self._state_to_game_state(state)
        self.last_reward_debug = {}
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
        original_action_index = int(action_index)
        buttons, action_name = self._action_to_buttons(original_action_index)

        pre_state = self.game.get_state()
        pre_game_state = self._state_to_game_state(pre_state)

        depth_obs = None

        if pre_state is not None:
            depth_raw = getattr(pre_state, "depth_buffer", None)

            if depth_raw is not None:
                try:
                    depth_obs = self.native_packer.pack_depth(depth_raw)
                except Exception:
                    depth_obs = None

        advised_action, action_advice_reason, action_advice_debug = self.shared_logic.advise_action(
            action_name=action_name,
            depth_obs=depth_obs,
            previous_state=self.previous_game_state,
            current_state=pre_game_state,
            allow_override=True,
        )

        if advised_action != action_name:
            action_name = advised_action
            action_index = self._action_name_to_index(action_name)
            buttons, action_name = self._action_to_buttons(action_index)

        try:
            self.game.make_action(buttons, self.frame_skip)

        except Exception as e:
            print(f"[vizdoom_env] make_action failed: {e}")
            traceback.print_exc()

            obs = self._restart_game_after_crash()

            info = {
                "backend": "vizdoom",
                "action_name": action_name,
                "original_action_index": original_action_index,
                "advised_action": action_name,
                "action_advice_reason": "vizdoom_restart_after_crash",
                "action_advice_debug": action_advice_debug,
                "buttons": buttons,
                "last_action": None,
                "episode_time": 0,
                "game_vars": {},
                "reward_debug": {
                    "crash_recovery": True,
                },
            }

            # Return truncated=True so SB3 treats this like an episode boundary,
            # not a fatal process error.
            return obs, -1.0, False, True, info

        self.step_count += 1

        done = self.game.is_episode_finished()
        truncated = self.step_count >= self.max_episode_steps

        state = None if done else self.game.get_state()
        reward = self._compute_reward(action_name, state)

        obs = self._make_obs()

        info = {
            "backend": "vizdoom",
            "action_name": action_name,
            "original_action_index": original_action_index,
            "advised_action": action_name,
            "action_advice_reason": action_advice_reason,
            "action_advice_debug": action_advice_debug,
            "buttons": buttons,
            "last_action": self.game.get_last_action(),
            "episode_time": self.game.get_episode_time(),
            "game_vars": self._get_game_vars(state),
            "reward_debug": self.last_reward_debug,
        }


        if done:
            reward += float(self.game.get_total_reward())

        return obs, float(reward), bool(done), bool(truncated), info
    
    def _restart_game_after_crash(self):
        """
        Restart ViZDoom after an unexpected engine exit.

        This prevents one ViZDoom crash from killing the entire PPO run.
        """

        print("[vizdoom_env] ViZDoom exited unexpectedly. Restarting game...")

        try:
            if self.game is not None:
                self.game.close()
        except Exception:
            pass

        self.game = None
        self.step_count = 0

        self._init_game()
        self.game.new_episode()

        state = self.game.get_state()
        self.previous_game_state = self._state_to_game_state(state)
        self.shared_logic.reset_episode()
        self.last_reward_debug = {
            "crash_recovery": True,
        }

        return self._make_obs()

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
