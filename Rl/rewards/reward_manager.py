import json
import os

from detectors.flash_detector import FlashDetector
from detectors.door_detector import DoorDetector
from detectors.enemy_detector import EnemyDetector

# Path: src/Rl/coach_directives.json  (one level above this file's directory)
_DIRECTIVES_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "coach_directives.json")
)


class RewardManager:

    def __init__(self):
        self.total_reward = 0
        self.breakdown = {}
        self.flash_detector = FlashDetector()
        self.door_detector = DoorDetector()
        self.enemy_detector = EnemyDetector()
        self.previous_ammo = None
        self.previous_health = 100
        self._last_health_change = 0
        self.visited_states = set()
        self.recent_rotations = 0
        self._coach_multipliers = self._load_coach_multipliers()

    @staticmethod
    def _load_coach_multipliers() -> dict:
        """Load reward multipliers saved by ai_coach.py, if present."""
        if not os.path.exists(_DIRECTIVES_PATH):
            return {}
        try:
            with open(_DIRECTIVES_PATH) as f:
                data = json.load(f)
            mults = data.get("reward_multipliers", {})
            if mults:
                print(f"[RewardManager] Loaded {len(mults)} coach directives "
                      f"from {_DIRECTIVES_PATH}")
            return mults
        except Exception as exc:
            print(f"[RewardManager] Warning: could not load coach directives: {exc}")
            return {}

    def add(self, name, value):
        mult = self._coach_multipliers.get(name, 1.0)
        adjusted = value * mult
        self.total_reward += adjusted

        if name not in self.breakdown:
            self.breakdown[name] = 0

        self.breakdown[name] += adjusted


    # --- event helpers ---

    def enemy_killed(self):
        self.add("enemy_killed", +1)

    def player_damaged(self):
        self.add("player_damaged", -1)

    def level_completed(self):
        self.add("level_completed", +10)

    def misc_step(self):
        self.add("step_penalty", -0.005)

    def penalize_stuck(self, pixel_diff: float):
        """Penalize when the screen barely changes — agent is pressed against a wall."""
        if pixel_diff < 1.0:
            self.add("stuck_penalty", -0.5)

    def died(self):
        self.add("death_penalty", -5)


    # Doom-specific useful signals

    def picked_health(self):
        self.add("health_pickup", +0.2)

    def picked_ammo(self):
        self.add("ammo_pickup", +0.2)

    def moved_forward(self):
        self.add("movement_bonus", +0.02)

    def interacted(self):
        self.add("interaction_bonus", +0.5)

    def detect_ammo_usage(self, action, current_ammo):

        if self.previous_ammo is None:

            self.previous_ammo = current_ammo
            return


        ammo_spent = self.previous_ammo - current_ammo


        if action == "shoot" and ammo_spent > 0:

            # default penalty for shooting
            self.add("ammo_used", -0.03)


        self.previous_ammo = current_ammo

    def punish_wall_shooting(self, action):

        frame = self.enemy_detector.capture()

        enemy_visible = self.enemy_detector.detect_enemy_presence(frame)

        if action == "shoot" and not enemy_visible:

            self.add("wall_shot_penalty", -0.25)

    def detect_combat_events(self):

        frame = self.flash_detector.capture()

        if self.flash_detector.detect_enemy_damage(frame):
            self.add("enemy_hit_flash", +1.0)

        if self.flash_detector.detect_player_damage(frame):
            self.add("player_hit_flash", -1.2)

        if self.flash_detector.detect_pickup_flash(frame):
            self.add("pickup_flash", +0.3)

    def detect_door_event(self, last_action):

        if last_action == "use":
            if self.door_detector.detect_door_motion():
                self.add("door_opened", +2.5)

    def detect_enemy_visible(self, action):

        frame = self.enemy_detector.capture()

        if self.enemy_detector.detect_enemy_presence(frame):
            self.add("enemy_visible", +0.05)

            if action == "shoot":
                self.add("shoot_when_enemy_visible", +0.4)

    def reward_efficient_shooting(self, action):

        frame = self.enemy_detector.capture()

        if action == "shoot":
            if self.enemy_detector.detect_enemy_centered(frame):
                self.add("accurate_fire_bonus", +1.2)
            elif self.enemy_detector.detect_enemy_presence(frame):
                self.add("reasonable_fire_bonus", +0.4)

    def detect_retaliation_reward(self, action):

        frame = self.flash_detector.capture()

        if self.flash_detector.detect_player_damage(frame):
            if action == "shoot":
                self.add("retaliation_shot", +0.6)

    def detect_enemy_centering(self, action):

        frame = self.enemy_detector.capture()

        if self.enemy_detector.detect_enemy_centered(frame):
            self.add("enemy_centered", +0.15)

            if action == "shoot":
                self.add("accurate_shot_attempt", +0.75)

    def detect_enemy_distance_change(self, action):

        growth = self.enemy_detector.detect_enemy_size_growth()


        if growth > 25:

            # enemy approaching center view

            if action == "shoot":

                self.add("shoot_close_enemy_bonus", +1.1)

            else:

                self.add("enemy_close_prepare", +0.2)


        elif growth < -25:

            # enemy moving away

            if action == "move_forward":

                self.add("closing_distance_bonus", +0.4)

    def detect_enemy_motion_response(self, action):

        motion = self.enemy_detector.detect_enemy_motion_direction()


        if motion > 8:

            if action == "turn_right":

                self.add("track_enemy_right", +0.5)


        elif motion < -8:

            if action == "turn_left":

                self.add("track_enemy_left", +0.5)

    def detect_health_change(self, current_health=None):

        if current_health is not None:
            change = current_health - self.previous_health
            self.previous_health = current_health
            self._last_health_change = change
            return change

        return self._last_health_change
    
    def detect_healing_behavior(self):

        health_change = self.detect_health_change()

        if health_change > 0:

            self.add("found_health_pack", +2)
    
    def detect_retreat_behavior(self, action):

        health_change = self.detect_health_change()

        current_health = self.previous_health


        LOW_HEALTH_THRESHOLD = 40


        if current_health < LOW_HEALTH_THRESHOLD:

            if health_change < 0:

                if action in ["move_backward", "strafe_left", "strafe_right"]:

                    self.add("smart_retreat", +1.5)

                if action == "shoot":

                    self.add("reckless_low_health_shooting", -1)

    def detect_damage_escape(self, action):

        health_change = self.detect_health_change()


        if health_change < 0:

            if action in ["strafe_left", "strafe_right"]:

                self.add("damage_strafe_escape", +1)


            if action == "move_backward":

                self.add("damage_backward_escape", +1)

    def detect_health_pickup(self):

        health_change = self.detect_health_change()

        if health_change > 5:

            self.add("medkit_collected", +3)


    def detect_ammo_pickup(self, current_ammo=None):

        if current_ammo is None:
            current_ammo = getattr(self, 'previous_ammo', 50)

        if not hasattr(self, "previous_ammo"):

            self.previous_ammo = current_ammo
            return


        ammo_change = current_ammo - self.previous_ammo

        self.previous_ammo = current_ammo


        if ammo_change > 0:

            self.add("ammo_collected", +2)

    def detect_resource_priority_behavior(self, action):

        health = self.previous_health
        ammo = self.previous_ammo


        LOW_HEALTH = 35
        LOW_AMMO = 5


        if health < LOW_HEALTH:

            if action in ["move_forward", "strafe_left", "strafe_right"]:

                self.add("searching_for_medkit", +0.5)


        if ammo < LOW_AMMO:

            if action == "move_forward":

                self.add("searching_for_ammo", +0.5)

    def detect_resource_waste(self):

        health = self.previous_health
        ammo = self.previous_ammo


        if health > 90:

            self.add("unnecessary_medkit_collection", -1)


        if ammo > 40:

            self.add("unnecessary_ammo_collection", -0.5)

    def get_frame_signature(self):

        frame = self.flash_detector.capture()

        small_frame = frame[::20, ::20]

        signature = small_frame.mean()

        return int(signature)
    
    def detect_curiosity_exploration(self):

        signature = self.get_frame_signature()


        if signature not in self.visited_states:

            self.visited_states.add(signature)

            self.add("new_area_discovered", +2)


        else:

            self.add("revisited_area_penalty", -0.05)

    def reward_forward_exploration(self, action):

        if action == "move_forward":

            self.add("forward_exploration_bonus", +0.2)

    def penalize_rotation_loops(self, action):

        if not hasattr(self, "recent_rotations"):

            self.recent_rotations = 0


        if action in ["turn_left", "turn_right"]:

            self.recent_rotations += 1

        else:

            self.recent_rotations = 0


        if self.recent_rotations > 8:

            self.add("rotation_loop_penalty", -1)

    
    def enemy_in_crosshair(self, frame):

        h, w, _ = frame.shape

        # small box in center
        cx, cy = w // 2, h // 2
        box = frame[cy-20:cy+20, cx-20:cx+20]

        # detect "enemy-like" pixels (tune this later)
        red_pixels = (box[:, :, 0] > 150) & (box[:, :, 1] < 80)

        return red_pixels.mean() > 0.05


    def get_reward(self):
        r = self.total_reward
        self.total_reward = 0
        return r

    def calculate_reward(self, action, game_state):
        """Run all detectors and return the total reward for this step."""
        self.detect_combat_events()
        self.detect_enemy_visible(action)
        self.detect_door_event(action)
        self.detect_enemy_centering(action)
        self.detect_ammo_usage(action, game_state["ammo"])
        self.punish_wall_shooting(action)
        self.reward_efficient_shooting(action)
        self.detect_retaliation_reward(action)
        self.detect_enemy_distance_change(action)
        self.detect_enemy_motion_response(action)
        self.detect_health_change(game_state["health"])
        self.detect_retreat_behavior(action)
        self.detect_healing_behavior()
        self.detect_damage_escape(action)
        self.detect_health_pickup()
        self.detect_ammo_pickup(game_state["ammo"])
        self.detect_resource_priority_behavior(action)
        self.detect_resource_waste()
        self.detect_curiosity_exploration()
        self.reward_forward_exploration(action)
        self.penalize_rotation_loops(action)
        self.misc_step()
        return self.get_reward()


    def get_breakdown(self):
        return self.breakdown