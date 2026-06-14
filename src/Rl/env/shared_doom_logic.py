import math
from navigation.level_guides import get_route_zones


class SharedDoomLogic:
    """
    Backend-independent Doom training logic.

    This class should not know whether the game is running through:
    - Doom Retro
    - ViZDoom
    - screen capture
    - depth buffers
    - keyboard control

    It only sees a normalized game_state dictionary.

    Expected game_state keys:
        health
        armor
        ammo
        selected_weapon
        kill_count
        item_count
        x
        y
        z
        angle
        depth_info
        objects
        sectors
    """

    def __init__(self, tile_size=64, level_name="freedoom_e1m1"):
        self.tile_size = tile_size
        self.visited_tiles = set()
        self.best_route_stage = -1
        self.previous_health = None
        self.previous_ammo = None
        self.previous_kills = 0
        self.previous_items = 0
        self.stuck_counter = 0
        self.wall_contact_steps = 0
        self.escape_action = None
        self.escape_steps_remaining = 0
        self.curriculum_stage = 1
        self.curriculum_step_count = 0
        self.curriculum_stage_steps = 0
        self.stage_route_hits = 0
        self.stage_item_hits = 0
        self.stage_kill_hits = 0

        # Simple E1M1/Freedoom-style route guide.
        # This is intentionally lightweight and should later move to navigation/level_guides.py.
        self.level_name = level_name
        self.route_zones = get_route_zones(level_name)
        
    def reset_episode(self):
        self.visited_tiles.clear()
        self.best_route_stage = -1
        self.previous_health = None
        self.previous_ammo = None
        self.previous_kills = 0
        self.previous_items = 0
        self.stuck_counter = 0
        self.wall_contact_steps = 0
        self.escape_action = None
        self.escape_steps_remaining = 0
        self.curriculum_stage_steps = 0

    def safe_float(self, value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def tile_key(self, x, y):
        x = self.safe_float(x)
        y = self.safe_float(y)

        return (
            int(x // self.tile_size),
            int(y // self.tile_size),
        )

    def distance_between(self, old_state, new_state):
        if old_state is None or new_state is None:
            return 0.0

        old_x = old_state.get("x")
        old_y = old_state.get("y")
        new_x = new_state.get("x")
        new_y = new_state.get("y")

        if old_x is None or old_y is None or new_x is None or new_y is None:
            return 0.0

        dx = self.safe_float(new_x) - self.safe_float(old_x)
        dy = self.safe_float(new_y) - self.safe_float(old_y)

        return math.sqrt(dx * dx + dy * dy)

    def tile_exploration_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        tile = self.tile_key(x, y)

        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            return 0.10

        return 0.0

    def movement_reward(self, previous_state, current_state, action_name):
        distance = self.distance_between(previous_state, current_state)

        if action_name == "move_forward":
            if distance > 6.0:
                return 0.06
            if distance > 2.0:
                return 0.03
            return -0.02

        if action_name in ["strafe_left", "strafe_right", "move_backward"]:
            if distance > 2.0:
                return 0.01

        if action_name in ["turn_left", "turn_right"]:
            return -0.001

        return 0.0

    def route_progress_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0, None

        x = self.safe_float(x)
        y = self.safe_float(y)

        reward = 0.0
        reached_name = None

        for idx, (name, target_x, target_y, radius, zone_reward) in enumerate(self.route_zones):
            dx = x - target_x
            dy = y - target_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= radius and idx > self.best_route_stage:
                self.best_route_stage = idx
                reward += zone_reward
                reached_name = name
                break

        return reward, reached_name

    def combat_reward(self, previous_state, current_state, action_name):
        reward = 0.0

        old_kills = self.safe_float(previous_state.get("kill_count") if previous_state else 0)
        new_kills = self.safe_float(current_state.get("kill_count"))

        old_items = self.safe_float(previous_state.get("item_count") if previous_state else 0)
        new_items = self.safe_float(current_state.get("item_count"))

        old_health = self.safe_float(previous_state.get("health") if previous_state else current_state.get("health"))
        new_health = self.safe_float(current_state.get("health"))

        old_ammo = self.safe_float(previous_state.get("ammo") if previous_state else current_state.get("ammo"))
        new_ammo = self.safe_float(current_state.get("ammo"))

        if new_kills > old_kills:
            reward += 1.0 * (new_kills - old_kills)

        if new_items > old_items:
            reward += 0.25 * (new_items - old_items)

        if new_health < old_health:
            reward -= 0.02 * (old_health - new_health)

        if action_name == "shoot" and new_ammo < old_ammo and new_kills <= old_kills:
            reward -= 0.01

        return reward

    def depth_region_info(self, depth_obs):
        """
        Creates simple wall/geometry info from a normalized 1-channel depth image.

        Expected depth_obs shape:
            (H, W) or (1, H, W)

        In our normalized depth:
            lower values are treated as closer
            higher values are treated as farther
        """

        if depth_obs is None:
            return {
                "front_blocked": False,
                "left_pressure": 0.0,
                "front_pressure": 0.0,
                "right_pressure": 0.0,
                "near_wall": False,
                "safest_escape_direction": None,
            }

        try:
            import numpy as np

            depth = np.asarray(depth_obs)

            if depth.ndim == 3:
                depth = depth[0]

            if depth.size == 0:
                raise ValueError("empty depth")

            h, w = depth.shape[:2]

            lower = depth[int(h * 0.45): int(h * 0.95), :]

            left = lower[:, : int(w * 0.33)]
            front = lower[:, int(w * 0.33): int(w * 0.66)]
            right = lower[:, int(w * 0.66):]

            # Since depth is normalized uint8, small values mean close.
            close_threshold = 70

            left_pressure = float((left < close_threshold).mean()) if left.size else 0.0
            front_pressure = float((front < close_threshold).mean()) if front.size else 0.0
            right_pressure = float((right < close_threshold).mean()) if right.size else 0.0

            front_blocked = front_pressure >= 0.45
            near_wall = max(left_pressure, front_pressure, right_pressure) >= 0.35

            pressures = {
                "turn_left": left_pressure,
                "turn_right": right_pressure,
            }

            # If left is blocked more, turning right is safer. If right is blocked more, turning left is safer.
            if left_pressure > right_pressure:
                safest_escape_direction = "turn_right"
            elif right_pressure > left_pressure:
                safest_escape_direction = "turn_left"
            else:
                safest_escape_direction = "turn_right" if front_blocked else None

            return {
                "front_blocked": bool(front_blocked),
                "left_pressure": left_pressure,
                "front_pressure": front_pressure,
                "right_pressure": right_pressure,
                "near_wall": bool(near_wall),
                "safest_escape_direction": safest_escape_direction,
            }

        except Exception:
            return {
                "front_blocked": False,
                "left_pressure": 0.0,
                "front_pressure": 0.0,
                "right_pressure": 0.0,
                "near_wall": False,
                "safest_escape_direction": None,
            }

    def depth_wall_reward(self, depth_info, action_name, distance_moved):
        if not depth_info:
            return 0.0

        reward = 0.0

        front_blocked = depth_info.get("front_blocked", False)
        near_wall = depth_info.get("near_wall", False)
        safest_escape_direction = depth_info.get("safest_escape_direction")

        if action_name == "move_forward" and front_blocked:
            if distance_moved < 1.0:
                reward -= 0.25
            else:
                reward -= 0.05

        if front_blocked and action_name in ["turn_left", "turn_right", "strafe_left", "strafe_right"]:
            reward += 0.04

        if safest_escape_direction is not None and action_name == safest_escape_direction:
            reward += 0.04

        if near_wall and action_name == "move_forward" and distance_moved < 1.0:
            reward -= 0.10

        return reward

    def use_action_reward(self, action_name, previous_state, current_state):
        if action_name != "use":
            return 0.0

        distance = self.distance_between(previous_state, current_state)

        if distance > 8.0:
            return 0.15

        return -0.02
    
    def advise_action(
        self,
        action_name,
        depth_obs=None,
        previous_state=None,
        current_state=None,
        allow_override=True,
    ):
        """
        Shared action advisor.

        This advisor should not fully control the agent.
        It only corrects obvious repeated bad behavior.

        Important:
        - Do not override forward movement immediately.
        - Give the agent a few attempts before correcting.
        - When escaping, persist one turn direction for several frames.
        """

        debug = {}

        if not allow_override:
            return action_name, "override_disabled", debug

        depth_info = self.depth_region_info(depth_obs)
        distance_moved = self.distance_between(previous_state, current_state)

        object_summary = self.summarize_objects(current_state or {})
        scene_info = self.infer_scene(
            current_state or {},
            depth_info=depth_info,
            object_summary=object_summary,
        )

        front_blocked = depth_info.get("front_blocked", False)
        near_wall = depth_info.get("near_wall", False)
        safest_escape_direction = depth_info.get("safest_escape_direction")

        debug["depth_info"] = depth_info
        debug["distance_moved"] = distance_moved
        debug["front_blocked"] = front_blocked
        debug["near_wall"] = near_wall
        debug["safest_escape_direction"] = safest_escape_direction
        debug["object_summary"] = object_summary
        debug["scene_info"] = scene_info

        # Track actual stuck behavior, not just visual depth pressure.
        if action_name == "move_forward" and distance_moved < 1.0:
            self.stuck_counter += 1
        elif distance_moved > 2.0:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        if front_blocked and distance_moved < 1.0:
            self.wall_contact_steps += 1
        elif distance_moved > 2.0:
            self.wall_contact_steps = max(0, self.wall_contact_steps - 2)
        else:
            self.wall_contact_steps = max(0, self.wall_contact_steps - 1)

        debug["stuck_counter"] = self.stuck_counter
        debug["wall_contact_steps"] = self.wall_contact_steps
        debug["escape_action"] = self.escape_action
        debug["escape_steps_remaining"] = self.escape_steps_remaining

        # If already escaping, persist the same turn direction.
        # This prevents left/right oscillation.
        if self.escape_steps_remaining > 0 and self.escape_action is not None:
            self.escape_steps_remaining -= 1
            return self.escape_action, "escape_persist", debug

        # Let the agent try forward first.
        # Do not immediately override just because depth says front_blocked.
        if action_name == "move_forward":
            if distance_moved > 2.0:
                return action_name, "forward_moving_ok", debug

            if self.stuck_counter < 3 and self.wall_contact_steps < 3:
                return action_name, "forward_trial_allowed", debug

        # Only override after repeated failed movement.
        serious_stuck = (
            self.stuck_counter >= 3
            or self.wall_contact_steps >= 3
        )

        if action_name == "move_forward" and front_blocked and serious_stuck:
            if safest_escape_direction is not None:
                self.escape_action = safest_escape_direction
            else:
                self.escape_action = "turn_right"

            self.escape_steps_remaining = 5

            return self.escape_action, "blocked_forward_escape", debug

        if serious_stuck and action_name in ["move_forward", "use"]:
            if safest_escape_direction is not None:
                self.escape_action = safest_escape_direction
            else:
                self.escape_action = "turn_right"

            self.escape_steps_remaining = 5

            return self.escape_action, "stuck_escape", debug
        
        if scene_info.get("scene_label") == "enemy_visible":
            if action_name == "move_forward":
                return "shoot", "enemy_visible_shoot", debug

        return action_name, "keep_action", debug
    
    def curriculum_weights(self):
        """
        Reward weights by curriculum stage.

        Stages:
            1 = basic movement/exploration
            2 = wall/depth avoidance
            3 = route progress
            4 = items/survival
            5 = full task/combat
        """

        if self.curriculum_stage == 1:
            return {
                "movement": 1.2,
                "exploration": 1.5,
                "wall": 0.5,
                "route": 0.0,
                "combat": 0.0,
                "use": 0.0,
            }

        if self.curriculum_stage == 2:
            return {
                "movement": 1.0,
                "exploration": 1.2,
                "wall": 1.5,
                "route": 0.3,
                "combat": 0.0,
                "use": 0.2,
            }

        if self.curriculum_stage == 3:
            return {
                "movement": 1.0,
                "exploration": 1.0,
                "wall": 1.2,
                "route": 1.5,
                "combat": 0.2,
                "use": 0.5,
            }

        if self.curriculum_stage == 4:
            return {
                "movement": 0.8,
                "exploration": 0.8,
                "wall": 1.0,
                "route": 1.2,
                "combat": 0.7,
                "use": 0.8,
            }

        return {
            "movement": 1.0,
            "exploration": 1.0,
            "wall": 1.0,
            "route": 1.0,
            "combat": 1.0,
            "use": 1.0,
        }

    def update_curriculum(self, debug):
        """
        Advances curriculum based on useful behavior.

        This is intentionally simple:
        - movement/exploration unlocks Stage 2
        - wall/contact behavior unlocks Stage 3
        - route progress unlocks Stage 4
        - item/combat progress unlocks Stage 5
        """

        self.curriculum_step_count += 1
        self.curriculum_stage_steps += 1

        route_zone = debug.get("route_zone")
        combat_reward = debug.get("combat_reward", 0.0)
        visited_tiles = debug.get("visited_tiles", 0)
        distance_moved = debug.get("distance_moved", 0.0)

        if route_zone is not None:
            self.stage_route_hits += 1

        if combat_reward >= 0.25:
            self.stage_item_hits += 1

        if combat_reward >= 1.0:
            self.stage_kill_hits += 1

        old_stage = self.curriculum_stage

        if self.curriculum_stage == 1:
            if visited_tiles >= 10 or self.curriculum_stage_steps >= 500:
                self.curriculum_stage = 2

        elif self.curriculum_stage == 2:
            if visited_tiles >= 18 or self.stage_route_hits >= 1 or self.curriculum_stage_steps >= 1000:
                self.curriculum_stage = 3

        elif self.curriculum_stage == 3:
            if self.stage_route_hits >= 3 or self.curriculum_stage_steps >= 1500:
                self.curriculum_stage = 4

        elif self.curriculum_stage == 4:
            if self.stage_item_hits >= 3 or self.stage_kill_hits >= 1 or self.curriculum_stage_steps >= 2000:
                self.curriculum_stage = 5

        advanced = self.curriculum_stage != old_stage

        if advanced:
            self.curriculum_stage_steps = 0
            print(f"[curriculum] advanced Stage {old_stage} -> {self.curriculum_stage}")

        return advanced
    
    def canonical_object_category(self, raw_name):
        """
        Convert backend-specific object names into shared categories.

        This protects the system from differences between:
        - Doom object names
        - Freedoom object names
        - ViZDoom labels
        - later Doom Retro predicted labels

        Returns one of:
            enemy
            item
            ammo
            health
            armor
            weapon
            barrel
            door
            switch
            key
            unknown
        """

        name = str(raw_name or "").lower()

        enemy_keywords = [
            "zombieman",
            "shotgunguy",
            "shotgun guy",
            "former human",
            "formerhuman",
            "imp",
            "demon",
            "spectre",
            "cacodemon",
            "baron",
            "hellknight",
            "lostsoul",
            "pain",
            "arachno",
            "mancubus",
            "revenant",
            "archvile",
            "cyberdemon",
            "spider",
            "enemy",
            "monster",
            "trooper",
            "sergeant",
            "beast",
            "minion",
            "possessed",
        ]

        health_keywords = [
            "stimpack",
            "stimpak",
            "medikit",
            "medkit",
            "health",
            "soulsphere",
            "soul sphere",
            "megasphere",
            "berserk",
        ]

        armor_keywords = [
            "armor",
            "armour",
            "greenarmor",
            "bluearmor",
            "securityarmor",
            "combatarmor",
        ]

        ammo_keywords = [
            "ammo",
            "clip",
            "clips",
            "shell",
            "shells",
            "rocket",
            "rockets",
            "cell",
            "cells",
            "backpack",
        ]

        weapon_keywords = [
            "shotgun",
            "supershotgun",
            "super shotgun",
            "chaingun",
            "rocketlauncher",
            "rocket launcher",
            "plasma",
            "bfg",
            "chainsaw",
            "weapon",
        ]

        key_keywords = [
            "key",
            "bluecard",
            "redcard",
            "yellowcard",
            "blueskull",
            "redskull",
            "yellowskull",
            "card",
            "skull",
        ]

        barrel_keywords = [
            "barrel",
            "explosive",
        ]

        door_keywords = [
            "door",
        ]

        switch_keywords = [
            "switch",
            "button",
            "lever",
        ]

        if any(keyword in name for keyword in enemy_keywords):
            return "enemy"

        if any(keyword in name for keyword in health_keywords):
            return "health"

        if any(keyword in name for keyword in armor_keywords):
            return "armor"

        if any(keyword in name for keyword in ammo_keywords):
            return "ammo"

        if any(keyword in name for keyword in weapon_keywords):
            return "weapon"

        if any(keyword in name for keyword in key_keywords):
            return "key"

        if any(keyword in name for keyword in barrel_keywords):
            return "barrel"

        if any(keyword in name for keyword in door_keywords):
            return "door"

        if any(keyword in name for keyword in switch_keywords):
            return "switch"

        return "unknown"
    
    def summarize_objects(self, game_state):
        """
        Backend-independent object summary.

        ViZDoom can provide true object metadata.
        Doom Retro can later provide predicted object labels.

        Returns a compact dictionary for reward/advice/debug.
        """

        objects = game_state.get("objects") or []

        summary = {
            "object_count": len(objects),
            "enemy_count": 0,
            "item_count_visible": 0,
            "ammo_count_visible": 0,
            "health_count_visible": 0,
            "armor_count_visible": 0,
            "weapon_count_visible": 0,
            "barrel_count_visible": 0,
            "door_like_count": 0,
            "nearest_enemy_distance": None,
            "nearest_item_distance": None,
            "visible_names": [],
            "unknown_count": 0,
            "categories": [],
        }

        px = self.safe_float(game_state.get("x"))
        py = self.safe_float(game_state.get("y"))

        enemy_keywords = [
            "zombieman",
            "shotgunguy",
            "imp",
            "demon",
            "spectre",
            "cacodemon",
            "baron",
            "lostsoul",
            "enemy",
            "monster",
            "trooper",
            "sergeant",
        ]

        item_keywords = [
            "stimpack",
            "medikit",
            "health",
            "armor",
            "bonus",
            "clip",
            "ammo",
            "shell",
            "rocket",
            "cell",
            "backpack",
            "key",
        ]

        weapon_keywords = [
            "shotgun",
            "chaingun",
            "launcher",
            "plasma",
            "bfg",
            "chainsaw",
        ]

        for obj in objects:
            name = str(obj.get("name", "")).lower()
            summary["visible_names"].append(name)

            ox = obj.get("position_x")
            oy = obj.get("position_y")

            dist = None
            if ox is not None and oy is not None:
                dx = self.safe_float(ox) - px
                dy = self.safe_float(oy) - py
                dist = math.sqrt(dx * dx + dy * dy)

            category = self.canonical_object_category(name)

            summary["categories"].append(category)

            if category == "unknown":
                summary["unknown_count"] += 1

            is_enemy = category == "enemy"
            is_item = category in ["health", "armor", "ammo", "key", "item"]
            is_weapon = category == "weapon"
            is_barrel = category == "barrel"
            is_door_like = category in ["door", "switch"]

            obj["category"] = category

            if is_enemy:
                summary["enemy_count"] += 1
                if dist is not None:
                    if summary["nearest_enemy_distance"] is None:
                        summary["nearest_enemy_distance"] = dist
                    else:
                        summary["nearest_enemy_distance"] = min(
                            summary["nearest_enemy_distance"],
                            dist,
                        )

            if is_item:
                summary["item_count_visible"] += 1
                if dist is not None:
                    if summary["nearest_item_distance"] is None:
                        summary["nearest_item_distance"] = dist
                    else:
                        summary["nearest_item_distance"] = min(
                            summary["nearest_item_distance"],
                            dist,
                        )

            if category == "ammo":
                summary["ammo_count_visible"] += 1

            if category == "health":
                summary["health_count_visible"] += 1

            if category == "armor":
                summary["armor_count_visible"] += 1

            if is_weapon:
                summary["weapon_count_visible"] += 1

            if is_barrel:
                summary["barrel_count_visible"] += 1

            if is_door_like:
                summary["door_like_count"] += 1

        # Keep debug readable.
        summary["visible_names"] = summary["visible_names"][:20]

        summary["categories"] = summary["categories"][:20]

        return summary

    def infer_scene(self, game_state, depth_info=None, object_summary=None):
        """
        Simple backend-independent scene label.

        This is not replacing your learned ScenePredictor.
        It gives ViZDoom a cheap scene summary and gives Doom Retro
        a common target format later.
        """

        if depth_info is None:
            depth_info = {}

        if object_summary is None:
            object_summary = self.summarize_objects(game_state)

        front_blocked = depth_info.get("front_blocked", False)
        near_wall = depth_info.get("near_wall", False)

        enemy_count = object_summary.get("enemy_count", 0)
        item_count_visible = object_summary.get("item_count_visible", 0)
        door_like_count = object_summary.get("door_like_count", 0)

        if enemy_count > 0:
            return {
                "scene_label": "enemy_visible",
                "scene_confidence": 0.85,
            }

        if door_like_count > 0:
            return {
                "scene_label": "door_or_switch",
                "scene_confidence": 0.75,
            }

        if front_blocked and near_wall:
            return {
                "scene_label": "front_blocked",
                "scene_confidence": 0.80,
            }

        if item_count_visible > 0:
            return {
                "scene_label": "item_visible",
                "scene_confidence": 0.70,
            }

        if near_wall:
            return {
                "scene_label": "near_wall",
                "scene_confidence": 0.65,
            }

        return {
            "scene_label": "open_space",
            "scene_confidence": 0.60,
        }

    def object_scene_reward(self, object_summary, scene_info, action_name):
        """
        Small shaping reward from shared object/scene perception.
        Keep this light so it does not overpower movement/route rewards.
        """

        reward = 0.0

        enemy_count = object_summary.get("enemy_count", 0)
        item_count_visible = object_summary.get("item_count_visible", 0)
        nearest_item_distance = object_summary.get("nearest_item_distance")
        scene_label = scene_info.get("scene_label")

        if scene_label == "enemy_visible":
            if action_name == "shoot":
                reward += 0.03
            elif action_name == "move_forward":
                reward -= 0.01

        if item_count_visible > 0 and nearest_item_distance is not None:
            if action_name == "move_forward":
                reward += 0.01

        if scene_label == "front_blocked" and action_name == "move_forward":
            reward -= 0.03

        return reward

    def compute_reward(
        self,
        previous_state,
        current_state,
        action_name,
        depth_obs=None,
    ):
        """
        Main shared reward function.

        Returns:
            reward, debug_info
        """

        reward = 0.0
        debug = {}

        distance_moved = self.distance_between(previous_state, current_state)
        depth_info = self.depth_region_info(depth_obs)

        object_summary = self.summarize_objects(current_state)
        scene_info = self.infer_scene(
            current_state,
            depth_info=depth_info,
            object_summary=object_summary,
        )

        living_cost = -0.001
        movement = self.movement_reward(previous_state, current_state, action_name)
        exploration = self.tile_exploration_reward(current_state)
        route_reward, route_zone = self.route_progress_reward(current_state)
        combat = self.combat_reward(previous_state, current_state, action_name)
        wall = self.depth_wall_reward(depth_info, action_name, distance_moved)
        use_reward = self.use_action_reward(action_name, previous_state, current_state)

        object_scene = self.object_scene_reward(
            object_summary=object_summary,
            scene_info=scene_info,
            action_name=action_name,
        )

        weights = self.curriculum_weights()

        reward += living_cost
        reward += movement * weights["movement"]
        reward += exploration * weights["exploration"]
        reward += route_reward * weights["route"]
        reward += combat * weights["combat"]
        reward += wall * weights["wall"]
        reward += use_reward * weights["use"]
        reward += object_scene

        debug["living_cost"] = living_cost
        debug["movement_reward"] = movement
        debug["exploration_reward"] = exploration
        debug["route_reward"] = route_reward
        debug["route_zone"] = route_zone
        debug["combat_reward"] = combat
        debug["wall_reward"] = wall
        debug["use_reward"] = use_reward
        debug["distance_moved"] = distance_moved
        debug["depth_info"] = depth_info
        debug["visited_tiles"] = len(self.visited_tiles)
        debug["curriculum_stage"] = self.curriculum_stage
        debug["curriculum_weights"] = weights
        debug["curriculum_stage_steps"] = self.curriculum_stage_steps
        debug["stage_route_hits"] = self.stage_route_hits
        debug["stage_item_hits"] = self.stage_item_hits
        debug["stage_kill_hits"] = self.stage_kill_hits
        debug["object_scene_reward"] = object_scene
        debug["object_summary"] = object_summary
        debug["scene_info"] = scene_info

        self.update_curriculum(debug)

        return float(reward), debug
