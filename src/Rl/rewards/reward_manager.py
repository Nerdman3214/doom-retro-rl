import numpy as np

from detectors.enemy_detector import EnemyDetector
from config.reward_weight_config import REWARD_WEIGHTS


class RewardManager:

    def __init__(self):
        self.enemy_detector = EnemyDetector()
        self.reset_episode()
        self.last_tile_reward_count = 0
        self.visited_tiles = set()

    def reset_episode(self):
        self.total_reward = 0.0
        self.breakdown = {}
        self.initial_distance = None
        self.closest_distance = None
        self.previous_distance_progress = 0.0
        self.initial_health = None
        self.initial_ammo = None
        self.visited_tiles = set()
        self.pixel_diff_history = []

    def reset(self):
        self.reset_episode()
        self.last_tile_reward_count = 0
        self.visited_tiles.clear()

    def add(self, name, value):
        self.total_reward += value
        if name not in self.breakdown:
            self.breakdown[name] = 0.0
        self.breakdown[name] += value

    # ------------------------------------------------------------------
    # FIX #8: Added movement_reward() — was called in doom_env but missing
    # ------------------------------------------------------------------
    def movement_reward(self, game_state: dict) -> float:
        """
        Reward real locomotion during the movement stage.
        Returns a float that doom_env adds directly to its reward accumulator.
        All stage-0 movement reward lives here to avoid double-counting.
        """
        reward = 0.0
        distance_moved = game_state.get("distance_moved", 0.0)
        action = game_state.get("action", "")

        movement_actions = [
            "move_forward", "move_backward",
            "turn_left", "turn_right",
            "strafe_left", "strafe_right",
        ]

        if distance_moved > 5.0:
            reward += 0.1   # meaningful physical movement
        elif action in movement_actions:
            reward -= 0.005  # tiny penalty for pressing move but not moving

        return reward

    def update_distance_reward(self, player_pos, goal_pos):
        if player_pos is None or goal_pos is None:
            return

        dist = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

        if self.initial_distance is None:
            self.initial_distance = dist

        if self.closest_distance is None:
            self.closest_distance = dist

        if dist < self.closest_distance:
            self.closest_distance = dist

        progress = 1.0 - (
            self.closest_distance / max(self.initial_distance, 1e-5)
        )
        progress = max(0.0, min(progress, 1.0))

        reward_delta = max(0.0, progress - self.previous_distance_progress)
        self.previous_distance_progress = progress
        self.add(
            "distance_progress",
            reward_delta * REWARD_WEIGHTS["distance_progress"],
        )

    def update_exploration_reward(self, player_pos):

        if player_pos is None:
            return

        tile = (
            int(player_pos[0] / 64),
            int(player_pos[1] / 64)
        )

        if tile not in self.visited_tiles:

            self.visited_tiles.add(tile)

            if len(self.visited_tiles) > self.last_tile_reward_count:

                self.add("exploration_tile", 0.05)

                self.last_tile_reward_count = len(self.visited_tiles)

    def detect_acid_damage(
        self,
        health_delta,
        distance_moved,
        floor_green_ratio
    ):

        if (
            health_delta < 0
            and distance_moved < 3
            and floor_green_ratio > 0.25
        ):
            return True

        return False

    def update_resource_reward(self, health, ammo):
        if health is None or ammo is None:
            return

        if self.initial_health is None:
            self.initial_health = max(health, 1)
        if self.initial_ammo is None:
            self.initial_ammo = max(ammo, 1)

        health_ratio = max(0.0, min(health / self.initial_health, 1.0))
        ammo_ratio = max(0.0, min(ammo / self.initial_ammo, 1.0))

        resource_score = (health_ratio + ammo_ratio) / 2.0
        self.add(
            "resource_efficiency",
            resource_score * REWARD_WEIGHTS["resource_efficiency"],
        )

    def update_stagnation_penalty(self, stuck_counter):
        stagnation = min(stuck_counter / 20.0, 1.0)
        self.add(
            "stagnation_penalty",
            -stagnation * REWARD_WEIGHTS["stagnation_penalty"],
        )

    def penalize_stuck(self, pixel_diff: float):
        """Track pixel changes and penalize minimal motion."""
        self.pixel_diff_history.append(pixel_diff)
        if len(self.pixel_diff_history) > 5:
            self.pixel_diff_history.pop(0)

        # FIX: reduced from -1.0 to -0.2 so it doesn't overwhelm other signals.
        # Termination in doom_env already handles hard stuck cases.
        if pixel_diff < 2.0:
            self.add("stuck_penalty", -0.2)

    def get_stuck_severity(self) -> float:
        """Return stuck severity (0=moving freely, 1.0=completely stuck)."""
        if not self.pixel_diff_history:
            return 0.0
        avg_diff = np.mean(self.pixel_diff_history)
        severity = max(0.0, 1.0 - (avg_diff / 2.0))
        return min(severity, 1.0)

    def reward_escape_behavior(self, action: str, pixel_diff: float):
        """Reward actions that increase pixel_diff when stuck."""
        if len(self.pixel_diff_history) < 2:
            return

        prev_avg = (
            np.mean(self.pixel_diff_history[:-1])
            if len(self.pixel_diff_history) > 1
            else 0.0
        )

        if pixel_diff > prev_avg + 0.01:
            if action in ["move_backward", "turn_left", "turn_right"]:
                self.add("escape_action", +0.03)

    def excessive_turn(self):
        self.add("turn_penalty", -0.02)

    def moved_forward(self):
        self.add("forward_bonus", +0.05)

    def enemy_visible(self):
        self.add("enemy_visible", +0.02)

    def enemy_centered(self):
        self.add("enemy_centered", +0.05)

    def smart_shot(self):
        self.add("smart_shot", +0.08)

    def wasted_shot(self):
        self.add("wasted_shot", -0.05)

    def get_reward(self):
        r = self.total_reward
        self.total_reward = 0.0
        return r

    def enemy_in_crosshair(self, frame):
        if frame is None:
            return False

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        box = frame[cy - 20:cy + 20, cx - 20:cx + 20]
        red_pixels = (box[:, :, 0] > 150) & (box[:, :, 1] < 80)
        return red_pixels.mean() > 0.05

    def calculate_reward(self, action, game_state):
        return self.get_reward()

    def get_breakdown(self):
        return self.breakdown