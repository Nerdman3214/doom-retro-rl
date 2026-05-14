import math


class CheckpointTracker:
    """
    Tracks route checkpoints and optional secret locations.

    This module rewards the agent for moving toward meaningful map goals
    instead of wandering randomly.

    It works best when game_state has real x/y coordinates from shared memory.
    If x/y are missing, it safely returns zero reward.
    """

    def __init__(self, reach_radius=96.0):
        self.reach_radius = reach_radius

        self.checkpoints = []
        self.secrets = []

        self.current_checkpoint_index = 0
        self.reached_checkpoints = set()
        self.reached_secrets = set()

        self.prev_checkpoint_distance = None
        self.prev_secret_distance = None

    def reset(self):
        self.current_checkpoint_index = 0
        self.reached_checkpoints.clear()
        self.reached_secrets.clear()
        self.prev_checkpoint_distance = None
        self.prev_secret_distance = None

    def set_level_guide(self, checkpoints=None, secrets=None):
        """
        checkpoints/secrets should be lists of dictionaries like:

        {
            "name": "start hallway",
            "x": 100.0,
            "y": 200.0,
            "reward": 5.0
        }
        """
        self.checkpoints = checkpoints or []
        self.secrets = secrets or []
        self.reset()

    def _player_position(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return None

        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None

    def _distance(self, pos, target):
        px, py = pos
        tx = float(target["x"])
        ty = float(target["y"])

        return math.sqrt((px - tx) ** 2 + (py - ty) ** 2)

    def _active_checkpoint(self):
        if self.current_checkpoint_index >= len(self.checkpoints):
            return None

        return self.checkpoints[self.current_checkpoint_index]

    def update(self, game_state):
        """
        Returns:
            reward, info

        info includes:
            distance_to_checkpoint
            checkpoint_name
            checkpoint_reached
            distance_to_secret
            nearest_secret_name
            secret_reached
        """
        reward = 0.0

        info = {
            "distance_to_checkpoint": None,
            "checkpoint_name": None,
            "checkpoint_reached": False,
            "distance_to_secret": None,
            "nearest_secret_name": None,
            "secret_reached": False,
            "all_checkpoints_reached": False,
        }

        pos = self._player_position(game_state)

        if pos is None:
            return reward, info

        # -----------------------------------------------------
        # Main route checkpoint
        # -----------------------------------------------------

        checkpoint = self._active_checkpoint()

        if checkpoint is not None:
            dist = self._distance(pos, checkpoint)

            info["distance_to_checkpoint"] = dist
            info["checkpoint_name"] = checkpoint.get("name", "checkpoint")

            if self.prev_checkpoint_distance is not None:
                progress = self.prev_checkpoint_distance - dist

                # Reward moving closer.
                if progress > 1.0:
                    reward += min(progress * 0.01, 0.15)

                # Penalize drifting away a little.
                elif progress < -5.0:
                    reward -= 0.03

            self.prev_checkpoint_distance = dist

            if dist <= checkpoint.get("radius", self.reach_radius):
                idx = self.current_checkpoint_index

                if idx not in self.reached_checkpoints:
                    reached_reward = checkpoint.get("reward", 5.0)
                    reward += reached_reward
                    self.reached_checkpoints.add(idx)
                    info["checkpoint_reached"] = True

                self.current_checkpoint_index += 1
                self.prev_checkpoint_distance = None

        if self.current_checkpoint_index >= len(self.checkpoints) and self.checkpoints:
            info["all_checkpoints_reached"] = True

        # -----------------------------------------------------
        # Optional secret tracking
        # -----------------------------------------------------

        nearest_secret = None
        nearest_secret_index = None
        nearest_secret_dist = None

        for i, secret in enumerate(self.secrets):
            if i in self.reached_secrets:
                continue

            dist = self._distance(pos, secret)

            if nearest_secret_dist is None or dist < nearest_secret_dist:
                nearest_secret = secret
                nearest_secret_index = i
                nearest_secret_dist = dist

        if nearest_secret is not None:
            info["distance_to_secret"] = nearest_secret_dist
            info["nearest_secret_name"] = nearest_secret.get("name", "secret")

            if self.prev_secret_distance is not None:
                progress = self.prev_secret_distance - nearest_secret_dist

                # Smaller reward than main checkpoint.
                if progress > 1.0:
                    reward += min(progress * 0.005, 0.08)

            self.prev_secret_distance = nearest_secret_dist

            if nearest_secret_dist <= nearest_secret.get("radius", self.reach_radius):
                reward += nearest_secret.get("reward", 10.0)
                self.reached_secrets.add(nearest_secret_index)
                self.prev_secret_distance = None
                info["secret_reached"] = True

        return reward, info