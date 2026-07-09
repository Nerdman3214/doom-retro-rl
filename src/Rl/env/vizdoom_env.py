
# Recovery prior imports
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path as _RecoveryPath

from sensory.world_state import build_world_state
from navigation.mission_plan import MissionTracker, get_freedoom1_e1m1_mission
from rewards.doom_brain_reward import compute_doom_brain_reward
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import traceback

from rewards.reward_manager import RewardManager
from director.route_director import RouteDirector
from navigation.level_guides import get_level_guide
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



class RecoveryActionPriorAdvisor:
    RECOVERY_ALLOWED = {
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
    }

    def __init__(self, checkpoint_path, device=None):

        # Recovery action prior: used only when stuck/wall-blocked.
        self.use_recovery_action_prior = True
        self.recovery_prior_min_confidence = 0.20
        self.recovery_prior_reward_scale = 0.04
        self.recovery_prior_mismatch_penalty = -0.004
        self.recovery_prior_debug_every = 100
        self.recovery_prior_call_count = 0
        self.recovery_action_prior = RecoveryActionPriorAdvisor(
            _RecoveryPath(__file__).resolve().parents[1] / "checkpoints" / "recovery_action_prior.pt"
        )
        self.checkpoint_path = _RecoveryPath(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.action_names = []
        self.ready = False

        if not self.checkpoint_path.exists():
            print(f"[recovery_prior] missing {self.checkpoint_path}")
            return

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.action_names = list(ckpt.get("action_names", []))
            num_actions = int(ckpt.get("num_actions", len(self.action_names) or 8))

            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions),
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.ready = True

            print(
                f"[recovery_prior] loaded {self.checkpoint_path} "
                f"num_actions={num_actions} device={self.device}"
            )
        except Exception as e:
            print(f"[recovery_prior] failed to load {self.checkpoint_path}: {e}")
            self.ready = False

    def preprocess(self, frame):
        if frame is None:
            return None

        arr = np.asarray(frame)

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3:
            return None

        if arr.shape[-1] > 3:
            arr = arr[..., :3]

        arr = arr.astype(np.uint8)

        import cv2
        arr = cv2.resize(arr, (84, 84), interpolation=cv2.INTER_AREA)
        arr = arr.astype(np.float32) / 255.0

        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def advise(self, frame):
        if not self.ready or self.model is None:
            return None, 0.0

        x = self.preprocess(frame)
        if x is None:
            return None, 0.0

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        ranked = torch.argsort(probs, descending=True).detach().cpu().tolist()

        for idx in ranked:
            if idx < 0 or idx >= len(self.action_names):
                continue

            name = self.action_names[idx]
            conf = float(probs[idx].detach().cpu().item())

            # Important: never use shoot/use as recovery advice.
            if name in self.RECOVERY_ALLOWED:
                return name, conf

        return None, 0.0


def recovery_prior_is_recovery_state(env):
    wall_contact_steps = int(getattr(env, "wall_contact_steps", 0) or 0)
    corner_trap_steps = int(getattr(env, "corner_trap_steps", 0) or 0)
    stuck_steps = int(getattr(env, "stuck_steps", 0) or 0)
    no_progress_steps = int(getattr(env, "no_progress_steps", 0) or 0)
    recovery_mode = bool(getattr(env, "recovery_mode", False))

    if recovery_mode:
        return True
    if wall_contact_steps >= 8:
        return True
    if corner_trap_steps >= 6:
        return True
    if stuck_steps >= 10:
        return True
    if no_progress_steps >= 18:
        return True

    return False


def recovery_action_prior_reward(env, frame, action_name):
    if not bool(getattr(env, "use_recovery_action_prior", False)):
        return 0.0

    if not recovery_prior_is_recovery_state(env):
        return 0.0

    advisor = getattr(env, "recovery_action_prior", None)
    if advisor is None or not getattr(advisor, "ready", False):
        return 0.0

    advised_action, confidence = advisor.advise(frame)

    env.recovery_prior_call_count = int(getattr(env, "recovery_prior_call_count", 0)) + 1

    if advised_action is None:
        return 0.0

    min_conf = float(getattr(env, "recovery_prior_min_confidence", 0.20))

    if confidence < min_conf:
        reward = 0.0
    elif str(action_name) == str(advised_action):
        reward = float(getattr(env, "recovery_prior_reward_scale", 0.04))
    else:
        reward = float(getattr(env, "recovery_prior_mismatch_penalty", -0.004))

    every = int(getattr(env, "recovery_prior_debug_every", 100))
    if every > 0 and env.recovery_prior_call_count % every == 0:
        print(
            f"[recovery_prior_reward] call={env.recovery_prior_call_count} "
            f"advised={advised_action} action={action_name} "
            f"conf={confidence:.2f} reward={reward:.4f} "
            f"wall={getattr(env, 'wall_contact_steps', 0)} "
            f"stuck={getattr(env, 'stuck_steps', 0)} "
            f"noprog={getattr(env, 'no_progress_steps', 0)}"
        )

    return float(reward)


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
        self.reward_manager = RewardManager()
        self.route_director = RouteDirector()

        self.current_level_name = "freedoom1_e1m1"
        self.level_guide = get_level_guide(self.current_level_name)


        self.last_progress_position = None
        self.no_position_change_steps = 0
        self.last_distance_to_goal = None
        self.no_distance_progress_steps = 0

        self.previous_position = None
        self.previous_health = None
        self.previous_ammo = None
        self.previous_kill_count = 0
        self.previous_item_count = 0

        self.route_progress_level = 0
        self.visited_tiles = set()
        self.previous_game_state = None
        self.last_reward_debug = {}
        # -----------------------------------------------------
        # Reward-only training mode
        # -----------------------------------------------------
        # In this mode, PPO chooses the action.
        # Helpers may add rewards/debug info, but they must not replace actions.
        self.reward_only_mode = True
        self.disable_action_helpers = True
        self.disable_fast_enemy_reaction = True
        self.disable_recovery_action_override = True
        self.disable_advisor_action_override = True
        self.disable_route_action_override = True

        self.game = None
        self.step_count = 0
        self.mission_tracker = MissionTracker(get_freedoom1_e1m1_mission())
        self.enable_doom_brain_reward = True
        self.last_health = None
        self.last_ammo = None
        self.last_kill_count = 0
        self.last_item_count = 0
        self.route_zones_reached = set()
        self.route_progress_level = 0
        self.curriculum_stage = 0
        self.max_stage = 4
        self.stage_success_counts = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }

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

        # Force Doom skill before game.init(). 1=easiest, 3=normal.


        try:


            skill = int(os.environ.get('VIZDOOM_SKILL', '1'))


            skill = max(1, min(5, skill))


            game.set_doom_skill(skill)


            print(f'[vizdoom_env] doom_skill={skill} (1=easiest, 3=normal)')


        except Exception as e:


            print(f'[vizdoom_env] could not set doom skill: {e}')


        game.init()
        self.game = game

    def get_stage_config(self):
        configs = {
            0: {
                "name": "movement_basic",
                "allow_shoot": False,
                "allow_use": False,
            },
            1: {
                "name": "movement_escape",
                "allow_shoot": False,
                "allow_use": False,
            },
            2: {
                "name": "doors_and_use",
                "allow_shoot": False,
                "allow_use": True,
            },
            3: {
                "name": "complete_level_basic",
                "allow_shoot": True,
                "allow_use": True,
            },
        }

        return configs.get(self.curriculum_stage, configs[3])


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

        return [
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "strafe_left",
            "strafe_right",
            "shoot",
            "use",
        ]
    
    def _angle_diff_degrees(self, a, b):
        diff = (a - b + 180.0) % 360.0 - 180.0
        return diff


    def _is_enemy_object(self, name):
        if not name:
            return False

        name = str(name).lower()

        enemy_keywords = [
            "zombie",
            "shotgun",
            "imp",
            "demon",
            "spectre",
            "cacodemon",
            "baron",
            "hell",
            "trooper",
            "sergeant",
            "former",
            "chaingun",
            "revenant",
            "mancubus",
            "arachnotron",
            "pain",
            "lost",
        ]

        return any(word in name for word in enemy_keywords)
    
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
    
    def route_progress_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        reward = 0.0
        route_zones = self.level_guide.get("route_zones", [])

        for zone in route_zones:
            name = zone.get("name")
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            radius = float(zone.get("radius", 128.0))
            zone_reward = float(zone.get("reward", 0.5))

            if name in self.route_zones_reached:
                continue

            dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5

            if dist <= radius:
                self.route_zones_reached.add(name)
                self.route_progress_level += 1
                reward += zone_reward
                self.reward_manager.add(f"viz_route_progress_{name}", zone_reward)

                print(
                    f"[viz_route_progress] reached={name} "
                    f"level={self.route_progress_level} "
                    f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
                )

        return reward

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


    def enrich_game_state(self, game_state):
        """
        Add DoomEnv-like combat/navigation keys to ViZDoom state.
        """

        x = game_state.get("x")
        y = game_state.get("y")
        angle = game_state.get("angle")

        enemy_visible = False
        enemy_centered = False
        enemy_left = False
        enemy_right = False
        enemy_distance = None
        enemy_count = 0

        if x is not None and y is not None and angle is not None:
            px = float(x)
            py = float(y)
            player_angle = float(angle)

            closest_dist = None
            closest_angle_diff = None

            for obj in game_state.get("objects", []):
                name = obj.get("name")

                if not self._is_enemy_object(name):
                    continue

                ox = obj.get("position_x")
                oy = obj.get("position_y")

                if ox is None or oy is None:
                    continue

                enemy_count += 1

                dx = float(ox) - px
                dy = float(oy) - py
                dist = math.sqrt(dx * dx + dy * dy)

                # Doom angle convention can vary, but this is good enough for teacher labels.
                target_angle = math.degrees(math.atan2(dy, dx))
                angle_diff = self._angle_diff_degrees(target_angle, player_angle)

                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_angle_diff = angle_diff

            if closest_dist is not None:
                closest_tiles = closest_dist / 64.0

                # Only count as visible/relevant if inside useful combat range.
                if closest_tiles <= 8.0 and abs(closest_angle_diff) <= 70.0:
                    enemy_visible = True
                    enemy_distance = closest_dist

                    if abs(closest_angle_diff) <= 10.0:
                        enemy_centered = True
                    elif closest_angle_diff < 0:
                        enemy_left = True
                    else:
                        enemy_right = True

        game_state["enemy_visible"] = enemy_visible
        game_state["enemy_centered"] = enemy_centered
        game_state["enemy_left"] = enemy_left
        game_state["enemy_right"] = enemy_right
        game_state["enemy_count"] = enemy_count
        game_state["enemy_distance"] = enemy_distance
        game_state["enemy_distance_tiles"] = (
            enemy_distance / 64.0 if enemy_distance is not None else None
        )

        game_state["curriculum_stage"] = self.curriculum_stage

        return game_state
    
    def fast_enemy_reaction_action(self, action, *args, **kwargs):
        """
        Reward-only mode:
        Do not override PPO action.
        Combat behavior should be learned through rewards.
        """
        return action


    def main_goal_progress_reward(self, director_result):
        """
        Reward getting closer to the real E1M1 exit.

        Uses RouteDirector output:
        - distance
        - distance_delta
        - hint

        Positive distance_delta means the agent moved closer to the true exit.
        """

        if director_result is None:
            return 0.0

        distance = director_result.get("distance")
        distance_delta = float(director_result.get("distance_delta", 0.0) or 0.0)
        hint = director_result.get("hint")

        if distance is None:
            return 0.0

        reward = 0.0

        if not hasattr(self, "best_main_goal_distance"):
            self.best_main_goal_distance = None

        if self.best_main_goal_distance is None:
            self.best_main_goal_distance = float(distance)

        if distance_delta > 1.0:
            progress_reward = min(0.12, distance_delta / 96.0)
            reward += progress_reward
            self.reward_manager.add("main_goal_closer", progress_reward)

        if float(distance) < self.best_main_goal_distance - 8.0:
            self.best_main_goal_distance = float(distance)
            reward += 0.10
            self.reward_manager.add("main_goal_new_best", 0.10)

        if distance_delta < -6.0:
            penalty = min(0.12, abs(distance_delta) / 96.0)
            reward -= penalty
            self.reward_manager.add("main_goal_moving_away", -penalty)

        if hint == "recover_unstuck":
            reward -= 0.08
            self.reward_manager.add("main_goal_stuck", -0.08)

        if hint == "goal_reached":
            reward += 10.0
            self.reward_manager.add("main_goal_reached", 10.0)

        return reward

    def goal_progress_reward(self, director_result):
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
            moved = math.sqrt((x - old_x) ** 2 + (y - old_y) ** 2)

            if moved < 2.0:
                self.no_position_change_steps += 1
            else:
                self.no_position_change_steps = 0

            if self.no_position_change_steps >= 4:
                penalty = min(0.30, 0.03 * self.no_position_change_steps)
                reward -= penalty
                self.reward_manager.add("no_position_change", -penalty)

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
                        penalty = min(0.25, 0.02 * self.no_distance_progress_steps)
                        reward -= penalty
                        self.reward_manager.add("no_goal_distance_progress", -penalty)

                self.last_distance_to_goal = distance

        return reward
    
    def combat_movement_reward(self, game_state, action_name, distance_moved):
        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        ammo = int(game_state.get("ammo", 0) or 0)
        health = int(game_state.get("health", 100) or 100)

        reward = 0.0
        moved = float(distance_moved or 0.0)

        if not enemy_visible:
            return 0.0

        if moved < 2.0 and action_name in ["turn_left", "turn_right", "shoot"]:
            reward -= 0.12
            self.reward_manager.add("combat_standing_still", -0.12)

        if action_name in ["strafe_left", "strafe_right"] and moved >= 2.0:
            reward += 0.08
            self.reward_manager.add("combat_strafe_movement", 0.08)

        if action_name == "move_backward":
            reward += 0.05
            self.reward_manager.add("combat_retreat_spacing", 0.05)

        if action_name == "shoot" and ammo <= 0:
            reward -= 0.60
            self.reward_manager.add("shoot_no_ammo", -0.60)

        elif action_name == "shoot" and enemy_centered and ammo > 0:
            reward += 0.30
            self.reward_manager.add("shoot_centered_with_ammo", 0.30)

        elif action_name == "shoot" and enemy_visible and not enemy_centered and ammo > 0:
            reward -= 0.15
            self.reward_manager.add("shoot_not_centered", -0.15)

        if health <= 35 and action_name in ["strafe_left", "strafe_right", "move_backward"]:
            reward += 0.10
            self.reward_manager.add("low_health_dodge", 0.10)

        return reward
    
    def _distance_moved(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        pos = (float(x), float(y))

        if self.previous_position is None:
            self.previous_position = pos
            return 0.0

        old_x, old_y = self.previous_position
        moved = math.sqrt((pos[0] - old_x) ** 2 + (pos[1] - old_y) ** 2)

        self.previous_position = pos

        return moved


    def sanitize_action(self, action_name, game_state):
        allowed = self.get_allowed_actions()

        if action_name not in allowed:
            return "move_forward" if "move_forward" in allowed else allowed[0]

        ammo = int(game_state.get("ammo", 0) or 0)
        enemy_visible = bool(game_state.get("enemy_visible", False))

        if action_name == "shoot":
            if "shoot" not in allowed:
                return "move_forward"

            if ammo <= 0:
                return "move_backward" if "move_backward" in allowed else "turn_right"

            if not enemy_visible:
                return "move_forward"

        if action_name == "use" and "use" not in allowed:
            return "move_forward"

        return action_name

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
    def update_curriculum_from_progress(self, game_state):
        reached = self.route_zones_reached

        if self.curriculum_stage == 0:
            if len(reached) >= 1:
                self.stage_success_counts[0] += 1

        elif self.curriculum_stage == 1:
            if False:
                self.stage_success_counts[1] += 1

        elif self.curriculum_stage == 2:
            if False:
                self.stage_success_counts[2] += 1

        elif self.curriculum_stage == 3:
            enemy_visible = bool(game_state.get("enemy_visible", False))
            kill_count = int(game_state.get("kill_count", 0) or 0)

            if enemy_visible or kill_count > 0:
                self.stage_success_counts[3] += 1

        current = self.curriculum_stage

        if (
            current < self.max_stage
            and self.stage_success_counts.get(current, 0) >= 3
        ):
            old = self.curriculum_stage
            self.curriculum_stage += 1
            print(f"[viz_curriculum] advanced Stage {old} -> {self.curriculum_stage}")
    
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
        ViZDoom real-map reward.

        This mirrors Doom Retro's real E1M1 guide:
        - reward progress toward actual exit (-400, 1296)
        - reward sequential route milestones
        - keep combat/item rewards
        - avoid old fake route names
        """

        reward = 0.0
        vars_now = self._get_game_vars(state)

        health = vars_now.get("health")
        kills = vars_now.get("kill_count", 0)
        items = vars_now.get("item_count", 0)
        x = vars_now.get("x")
        y = vars_now.get("y")

        # Small living cost.
        reward -= 0.001

        # Combat/items still matter.
        if self.last_health is not None and health is not None:
            health_delta = health - self.last_health
            if health_delta < 0:
                reward += health_delta * 0.02

        if kills > self.last_kill_count:
            reward += 1.0 * (kills - self.last_kill_count)

        if items > self.last_item_count:
            reward += 0.25 * (items - self.last_item_count)

        # Real map navigation.
        guide = getattr(self, "level_guide", {}) or {}
        goal = guide.get("main_goal") or {}

        if x is not None and y is not None and goal.get("x") is not None and goal.get("y") is not None:
            x = float(x)
            y = float(y)
            gx = float(goal["x"])
            gy = float(goal["y"])

            dx = x - gx
            dy = y - gy
            dist = (dx * dx + dy * dy) ** 0.5

            if self.best_main_goal_distance is None:
                self.best_main_goal_distance = dist
            else:
                improvement = self.best_main_goal_distance - dist

                if improvement > 0:
                    reward += min(0.10, improvement * 0.003)
                    self.best_main_goal_distance = min(self.best_main_goal_distance, dist)
                elif improvement < -24:
                    reward -= 0.02

            # Sequential route zones.
            route_zones = guide.get("route_zones", [])
            next_idx = int(getattr(self, "route_progress_level", 0))

            if next_idx < len(route_zones):
                zone = route_zones[next_idx]
                zx = float(zone.get("x", 0.0))
                zy = float(zone.get("y", 0.0))
                radius = float(zone.get("radius", 128.0))
                zone_reward = float(zone.get("reward", 0.05))
                name = zone.get("name", f"route_zone_{next_idx}")

                zdx = x - zx
                zdy = y - zy
                zdist = (zdx * zdx + zdy * zdy) ** 0.5

                if zdist <= radius:
                    self.route_progress_level = next_idx + 1
                    self.route_zones_reached.add(name)
                    reward += zone_reward

                    print(
                        f"[viz_route_progress] reached={name} "
                        f"level={self.route_progress_level} "
                        f"x={x:.1f} y={y:.1f} reward={zone_reward:.2f}"
                    )

            # Penalize old east/right drift.
            bounds = guide.get("safe_bounds") or {}
            max_x = float(bounds.get("max_x", 200.0))

            if x > max_x:
                reward -= 0.03
                if x > max_x + 500:
                    reward -= 0.08

        self.last_health = health
        self.last_ammo = vars_now.get("ammo")
        self.last_kill_count = kills
        self.last_item_count = items

        return float(reward)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.game.set_seed(int(seed))

        self.game.new_episode()
        self.step_count = 0
        if hasattr(self, "mission_tracker"):
            self.mission_tracker.reset()
        self.shared_logic.reset_episode()
        self.reward_manager.reset()
        self.route_director.reset()
        state = self.game.get_state()
        start_state = self._state_to_game_state(state)
        self.spawn_position = (
            float(start_state.get("x", 0.0) or 0.0),
            float(start_state.get("y", 0.0) or 0.0),
        )

        self.last_progress_position = None
        self.no_position_change_steps = 0
        self.last_distance_to_goal = None
        self.no_distance_progress_steps = 0
        self.previous_position = None
        self.visited_tiles.clear()
        self.last_exit_distance = None
        self.best_exit_distance = None

        state = self.game.get_state()
        self.previous_game_state = self._state_to_game_state(state)
        self.last_reward_debug = {}
        vars_now = self._get_game_vars(state)

        self.last_health = vars_now.get("health")
        self.last_ammo = vars_now.get("ammo")
        self.last_kill_count = vars_now.get("kill_count", 0)
        self.last_item_count = vars_now.get("item_count", 0)
        self.best_main_goal_distance = None
        self.route_progress_level = 0
        self.route_zones_reached = set()
        self.route_zones_reached = set()
        self.route_progress_level = 0
        self.goal_bubble_best_dist = {}
        self.goal_bubble_last_dist = {}
        self.goal_bubble_stall_steps = {}

        obs = self._make_obs()

        info = {
            "backend": "vizdoom",
            "game_vars": vars_now,
        }

        return obs, info
    
    def spawn_escape_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        x = float(x)
        y = float(y)

        sx, sy = getattr(self, "spawn_position", (0.0, 0.0))
        dist_from_spawn = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5

        reward = 0.0

        if self.curriculum_stage <= 1:
            if dist_from_spawn > 128:
                reward += 0.05
                self.reward_manager.add("spawn_escape_small", 0.05)

            if dist_from_spawn > 256:
                reward += 0.10
                self.reward_manager.add("spawn_escape_medium", 0.10)

            if dist_from_spawn > 384:
                reward += 0.20
                self.reward_manager.add("spawn_escape_large", 0.20)

        return reward

    def step(self, action_index):
        original_action_index = int(action_index)
        buttons, action_name = self._action_to_buttons(original_action_index)

        pre_state = self.game.get_state()
        pre_game_state = self._state_to_game_state(pre_state)
        pre_state = self.game.get_state()
        pre_game_state = self._state_to_game_state(pre_state)
        pre_game_state = self.enrich_game_state(pre_game_state)

        action_name = self.sanitize_action(action_name, pre_game_state)

        if self.curriculum_stage >= 3:
            before_fast = action_name
            action_name = self.fast_enemy_reaction_action(
                action_name=action_name,
                game_state=pre_game_state,
            )

            if action_name != before_fast and self.step_count % 25 == 0:
                print(f"[vizdoom] fast_enemy_reaction: {before_fast} -> {action_name}")

        action_index = self._action_name_to_index(action_name)
        buttons, action_name = self._action_to_buttons(action_index)

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

        # Recovery-prior reward only activates during stuck/wall-blocked states.
        try:
            recovery_frame = None
            if 'post_game_state' in locals() and post_game_state is not None:
                recovery_frame = getattr(post_game_state, 'screen_buffer', None)
            if recovery_frame is None and 'state' in locals() and state is not None:
                recovery_frame = getattr(state, 'screen_buffer', None)
            reward += recovery_action_prior_reward(self, recovery_frame, action_name)
        except Exception as e:
            if int(getattr(self, 'recovery_prior_call_count', 0)) % 500 == 0:
                print(f'[recovery_prior_reward] skipped due error: {e}')

        post_game_state = self._state_to_game_state(state)
        post_game_state = self.enrich_game_state(post_game_state)

        distance_moved = self._distance_moved(post_game_state)
        post_game_state["distance_moved"] = distance_moved
        post_game_state["action"] = action_name

        director_result = self.route_director.evaluate(
            game_state=post_game_state,
            action=action_name,
            level_guide=self.level_guide,
        )

        reward += self.spawn_escape_reward(post_game_state)
        reward += self.route_progress_reward(post_game_state)
        reward += self.main_goal_progress_reward(director_result)
        reward += self.goal_progress_reward(director_result)
        reward += self.stagnation_penalty(
            game_state=post_game_state,
            director_result=director_result,
        )
        reward += self.combat_movement_reward(
            game_state=post_game_state,
            action_name=action_name,
            distance_moved=distance_moved,
        )

        obs = self._make_obs()
        reward += self.route_progress_reward(post_game_state)
        reward += self.goal_attraction_bubble_reward(post_game_state)
        self.update_curriculum_from_progress(post_game_state)
        reward += self.exit_distance_progress_reward(post_game_state)

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
            "game_state": post_game_state,
            "director_result": director_result,
            "reward_debug": self.last_reward_debug,
        }


        if done:
            reward += float(self.game.get_total_reward())

        world_state = build_world_state(post_game_state, {})
        mission_update = self.mission_tracker.update(post_game_state, world_state)
        brain_reward, brain_events = compute_doom_brain_reward(
            game_state=post_game_state,
            action_name=action_name,
            world_state=world_state,
            mission_update=mission_update,
        )
        reward += brain_reward

        # Track wall/rail pushing or no-progress behavior.
        if not hasattr(self, "reroute_stuck_steps"):
            self.reroute_stuck_steps = 0

        exit_delta = float(director_result.get("distance_delta", 0.0) or 0.0)

        is_not_progressing = exit_delta < 1.0
        is_barely_moving = distance_moved < 2.0
        is_push_action = action_name in ["move_forward", "strafe_left", "strafe_right"]

        if is_push_action and is_not_progressing and is_barely_moving:
            self.reroute_stuck_steps += 1
        else:
            self.reroute_stuck_steps = 0

        if self.reroute_stuck_steps >= 8:
            reward -= 12.0
            info["reroute_needed"] = True

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
