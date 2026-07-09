import math


class DoomRuleEnforcer:
    """
    Unified Director + HelperBot rule system.

    This is the outside force / teacher.

    It does NOT need to control the agent directly.
    It tells the agent whether its chosen action follows Doom rules.

    Core skills:
    1. Navigation survival
    2. Basic fighting
    3. Navigation + fighting together
    4. Items/resources
    5. Doors/keys/use logic
    6. Secrets/advanced routes
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_position = None
        self.no_movement_steps = 0
        self.best_target_distance = None
        self.reached_rules = set()
        self.recent_actions = []

    def _neg(self, amount):
        """
        Test mode:
        make negative rewards much stronger.
        """
        return -abs(float(amount)) * 1.0

    def _pos(self, amount):
        """
        Keep positive rewards normal for this test.
        """
        return abs(float(amount))

    def _dist(self, ax, ay, bx, by):
        dx = float(ax) - float(bx)
        dy = float(ay) - float(by)
        return math.sqrt(dx * dx + dy * dy)

    def _movement_delta(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")

        if x is None or y is None:
            return 0.0

        pos = (float(x), float(y))

        if self.last_position is None:
            self.last_position = pos
            return 0.0

        moved = self._dist(pos[0], pos[1], self.last_position[0], self.last_position[1])
        self.last_position = pos

        if moved < 2.0:
            self.no_movement_steps += 1
        else:
            self.no_movement_steps = 0

        return moved

    def _current_target(self, game_state, mission_update=None, level_guide=None):
        """
        Exit-first mode.

        Do not use corridor, slope, post-corridor-room, route bubbles,
        or secret targets. The agent should adapt from experience, not be
        told that an early corridor zone is progress.

        Only target: real E1M1 exit area.
        """

        return {
            "name": "level_exit",
            "kind": "exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
            "hint": "complete_level",
        }

    def _target_distance_reward(self, game_state, target):
        if not target:
            return 0.0, []

        x = game_state.get("x")
        y = game_state.get("y")
        tx = target.get("x")
        ty = target.get("y")

        if x is None or y is None or tx is None or ty is None:
            return 0.0, []

        dist = self._dist(x, y, tx, ty)
        events = []
        reward = 0.0

        if self.best_target_distance is None:
            self.best_target_distance = dist

        improvement = self.best_target_distance - dist

        if improvement > 8.0:
            reward += min(8.0, improvement / 16.0)
            self.best_target_distance = dist
            events.append("navigation_closer_to_target")

        elif improvement < -24.0:
            reward += self._neg(5.0)
            events.append("navigation_wrong_way")

        return reward, events

    def _navigation_survival_rule(self, game_state, action_name, moved):
        reward = 0.0
        events = []

        front_blocked = bool(game_state.get("front_blocked", False))
        front_ratio = float(game_state.get("front_wall_ratio", 0.0) or 0.0)
        wall_bubble = game_state.get("wall_bubble_level")
        scene_label = game_state.get("scene_label")
        sensory = game_state.get("sensory_situation")

        obstacle = (
            front_blocked
            or front_ratio >= 0.50
            or wall_bubble in ["orange", "red"]
            or scene_label in ["front_wall", "obstacle", "boundary_or_stuck_wall", "half_wall_rail", "side_rail"]
            or sensory in ["front_blocked", "stuck_or_looping", "half_wall_rail", "side_rail"]
        )

        if obstacle and action_name == "move_forward":
            reward += self._neg(15.0)
            events.append("bad_push_into_obstacle")

        if obstacle and action_name in ["move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]:
            reward += 4.0
            events.append("good_obstacle_escape_action")

        if self.no_movement_steps >= 18:
            reward += self._neg(15.0)
            events.append("bad_no_movement_2sec")

        if moved > 5.0 and not obstacle:
            reward += 0.10
            events.append("good_safe_navigation_movement")

        return reward, events

    def _basic_fighting_rule(self, game_state, action_name):
        reward = 0.0
        events = []

        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        ammo = int(game_state.get("ammo", 0) or 0)
        kill_delta = int(game_state.get("kill_delta", 0) or 0)

        if enemy_visible:
            reward += 0.05
            events.append("enemy_awareness")

        if enemy_visible and enemy_centered:
            reward += 0.10
            events.append("enemy_centered")

        if action_name == "shoot":
            if enemy_visible and enemy_centered and ammo > 0:
                reward += 12.0
                events.append("good_shoot_centered_enemy")
            elif enemy_visible and ammo > 0:
                reward += 4.0
                events.append("okay_shoot_visible_enemy")
            elif ammo <= 0:
                reward += self._neg(5.0)
                events.append("bad_shoot_no_ammo")
            else:
                reward += self._neg(3.0)
                events.append("bad_blind_or_wasted_shot")

        if kill_delta > 0:
            reward += 10.0 * kill_delta
            events.append("enemy_kill")

        return reward, events

    def _navigation_fighting_rule(self, game_state, action_name, moved):
        reward = 0.0
        events = []

        enemy_visible = bool(game_state.get("enemy_visible", False))
        health_delta = int(game_state.get("health_delta", 0) or 0)

        if enemy_visible and action_name in ["strafe_left", "strafe_right", "move_backward"] and moved > 2.0:
            reward += 3.0
            events.append("good_combat_movement")

        if enemy_visible and action_name == "move_forward" and health_delta < 0:
            reward += self._neg(5.0)
            events.append("bad_walk_into_enemy_damage")

        if health_delta < 0 and action_name in ["strafe_left", "strafe_right", "move_backward"]:
            reward += 4.0
            events.append("good_dodge_when_damaged")

        return reward, events

    def _items_resources_rule(self, game_state, action_name):
        reward = 0.0
        events = []

        health = int(game_state.get("health", 100) or 100)
        ammo = int(game_state.get("ammo", 0) or 0)
        health_delta = int(game_state.get("health_delta", 0) or 0)
        ammo_delta = int(game_state.get("ammo_delta", 0) or 0)
        weapon_delta = int(game_state.get("weapon_delta", 0) or 0)

        pickup_visible = bool(game_state.get("pickup_visible", False))
        health_visible = bool(game_state.get("health_visible", False))
        ammo_visible = bool(game_state.get("ammo_visible", False))

        if health_delta > 0:
            reward += 10.0
            events.append("picked_health")

        if ammo_delta > 0:
            reward += 8.0
            events.append("picked_ammo")

        if weapon_delta > 0:
            reward += 15.0
            events.append("picked_weapon")

        if health < 40 and health_visible and action_name in ["move_forward", "strafe_left", "strafe_right"]:
            reward += 3.0
            events.append("move_toward_needed_health")

        if ammo <= 5 and ammo_visible and action_name in ["move_forward", "strafe_left", "strafe_right"]:
            reward += 3.0
            events.append("move_toward_needed_ammo")

        if pickup_visible and action_name in ["move_forward", "strafe_left", "strafe_right"]:
            reward += 1.0
            events.append("move_toward_pickup")

        return reward, events

    def _doors_keys_use_rule(self, game_state, action_name):
        reward = 0.0
        events = []

        door_visible = bool(game_state.get("door_visible", False))
        door_centered = bool(game_state.get("door_centered", False))
        near_use_point = bool(game_state.get("near_use_point", False))
        opened_door = bool(game_state.get("opened_door", False))
        scene_label = game_state.get("scene_label")

        door_like = (
            door_visible
            or near_use_point
            or scene_label in ["door_or_button", "switch", "locked_door"]
        )

        if action_name == "use":
            if door_like or door_centered:
                reward += 10.0
                events.append("good_use_near_door")
            else:
                reward += self._neg(2.0)
                events.append("bad_use_spam")

        if opened_door:
            reward += 20.0
            events.append("opened_door")

        return reward, events

    def _secrets_advanced_rule(self, game_state, action_name, mission_update=None):
        reward = 0.0
        events = []

        secret_reached = bool(game_state.get("secret_reached", False))
        level_complete = bool(game_state.get("level_complete", False))

        if secret_reached:
            reward += 25.0
            events.append("secret_reached")

        # Secrets/optional objectives disabled for exit-first training.
        # Add them back after the agent can complete E1M1.

        if level_complete:
            reward += 500.0
            events.append("level_complete")

        return reward, events

    def _clip_total_reward(self, reward):
        """
        Keep the 6x-negative test strong but bounded enough for PPO.
        """
        return max(-30.0, min(120.0, float(reward)))

    def preferred_action(self, game_state, target=None):
        """
        The old Director + HelperBot idea, but converted into advice.

        This is useful for debug, reward shaping, and later imitation.
        It should not automatically override PPO unless teacher_mode=True.
        """

        enemy_visible = bool(game_state.get("enemy_visible", False))
        enemy_centered = bool(game_state.get("enemy_centered", False))
        ammo = int(game_state.get("ammo", 0) or 0)
        front_blocked = bool(game_state.get("front_blocked", False))
        front_ratio = float(game_state.get("front_wall_ratio", 0.0) or 0.0)

        if front_blocked or front_ratio >= 0.55:
            return "turn_left"

        if enemy_visible and enemy_centered and ammo > 0:
            return "shoot"

        if enemy_visible and not enemy_centered:
            return "turn_left"

        if target:
            x = game_state.get("x")
            tx = target.get("x")
            if x is not None and tx is not None:
                if float(tx) < float(x) - 128:
                    return "turn_left"
                if float(tx) > float(x) + 128:
                    return "turn_right"

        return "move_forward"

    def evaluate(self, game_state, action_name, mission_update=None, level_guide=None):
        """
        Return a unified reward/debug packet.
        """

        moved = self._movement_delta(game_state)
        target = self._current_target(game_state, mission_update, level_guide)

        total = 0.0
        events = []

        parts = [
            self._navigation_survival_rule(game_state, action_name, moved),
            self._basic_fighting_rule(game_state, action_name),
            self._navigation_fighting_rule(game_state, action_name, moved),
            self._items_resources_rule(game_state, action_name),
            self._doors_keys_use_rule(game_state, action_name),
            self._secrets_advanced_rule(game_state, action_name, mission_update),
            self._target_distance_reward(game_state, target),
        ]

        breakdown = {}

        names = [
            "navigation_survival",
            "basic_fighting",
            "navigation_fighting",
            "items_resources",
            "doors_keys_use",
            "secrets_advanced",
            "target_distance",
        ]

        for name, (reward, evs) in zip(names, parts):
            total += float(reward)
            breakdown[name] = float(reward)
            events.extend(evs)

        advice = self.preferred_action(game_state, target)

        if action_name == advice:
            total += 0.5
            breakdown["advice_alignment"] = 0.5
            events.append("matched_rule_advice")
        else:
            breakdown["advice_alignment"] = 0.0

        total = self._clip_total_reward(total)

        return {
            "reward": float(total),
            "events": events,
            "breakdown": breakdown,
            "preferred_action": advice,
            "target": target,
            "no_movement_steps": self.no_movement_steps,
            "moved": moved,
        }
