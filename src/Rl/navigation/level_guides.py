"""
Manual level guide database.

This file stores map-specific knowledge:
- main goal / exit
- optional secrets
- keycards
- locked doors
- switches/buttons
- elevators/doors/use points

Important:
The coordinates are still placeholders until you record real x/y values
from your debug logs.

The important architecture change is:
RouteDirector should read goals from here instead of hardcoding one map path.
"""

DEFAULT_EMPTY_GUIDE = {
    "name": "generic",
    "main_goal": None,
    "route_nodes": [],
    "route_zones": [],
    "checkpoints": [],
    "secrets": [],
    "keys": [],
    "locked_doors": [],
    "switches": [],
    "use_points": [],
    "strategies": [],
}


LEVEL_GUIDES = {
    "freedoom1_e1m1": {
        "name": "freedoom1_e1m1",

        "route_nodes": [
            {
                "name": "spawn_exit",
                "x": 530,
                "y": 300,
                "radius": 96,
                "type": "milestone",
                "next": ["right_route", "central_route"],
                "reward": 0.5,
            },
            {
                "name": "right_route",
                "x": 570,
                "y": 304,
                "radius": 128,
                "type": "route",
                "next": ["door_area"],
                "reward": 0.8,
            },
            {
                "name": "door_area",
                "x": 636,
                "y": 304,
                "radius": 128,
                "type": "use_area",
                "hint_action": "use",
                "next": ["combat_corridor"],
                "reward": 1.0,
            },
            {
                "name": "combat_corridor",
                "x": 720,
                "y": 316,
                "radius": 144,
                "type": "combat_route",
                "next": ["exit_route", "resource_recovery"],
                "reward": 1.5,
            },
            {
                "name": "exit_route",
                "x": 930,
                "y": 464,
                "radius": 160,
                "type": "exit_path",
                "next": ["level_exit"],
                "reward": 2.0,
            },
            {
                "name": "blue_key_route",
                "type": "key_route",
                "x": 400,
                "y": 300,
                "radius": 160,
                "provides_key": "blue",
                "next": ["blue_door"],
                "reward": 2.0,
            },
            {
                "name": "non_blue_left_route",
                "type": "alternate_route",
                "x": 300,
                "y": 500,
                "radius": 160,
                "next": ["exit_route"],
                "reward": 1.5,
            },
            {
                "name": "blue_door",
                "type": "locked_door",
                "requires_key": "blue",
                "x": 900,
                "y": 460,
                "radius": 128,
                "next": ["exit_route"],
                "reward": 2.0,
            },
        ],

        "route_zones": [],

        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": 1200,
            "y": 464,
            "radius": 160,
            "reward": 10.0,
        },

        "checkpoints": [],
        "secrets": [],
        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],

        "strategies": [
            "safe_main_route",
            "combat_clear_then_exit",
            "resource_recovery",
            "explore_if_lost",
        ],
    }
}

GENERIC_EXPLORATION_ROUTE_ZONES = []


def normalize_level_name(level_name=None):
    normalized = str(level_name or "freedoom1_e1m1").lower()

    aliases = {
        "freedoom_e1m1": "freedoom1_e1m1",
        "freedoom1_e1m1": "freedoom1_e1m1",
        "e1m1": "freedoom1_e1m1",
        "doom_e1m1": "freedoom1_e1m1",
        "generic": "generic",
    }

    return aliases.get(normalized, normalized)


def get_level_guide(level_name=None):
    normalized = normalize_level_name(level_name)

    guide = LEVEL_GUIDES.get(normalized)

    if guide is None:
        return DEFAULT_EMPTY_GUIDE.copy()

    fixed = DEFAULT_EMPTY_GUIDE.copy()
    fixed.update(guide)
    return fixed


def get_route_zones(level_name=None):
    """
    Return route zones for a known level.

    Supports both dictionary route zones and tuple route zones.
    SharedDoomLogic expects route zones as:
        (name, x, y, radius, reward)
    """

    normalized = normalize_level_name(level_name)

    if normalized == "generic":
        return list(GENERIC_EXPLORATION_ROUTE_ZONES)

    guide = get_level_guide(normalized)
    route_zones = guide.get("route_zones", [])

    converted = []

    for zone in route_zones:
        if isinstance(zone, dict):
            converted.append(
                (
                    zone.get("name", "unnamed_zone"),
                    zone.get("x", 0),
                    zone.get("y", 0),
                    zone.get("radius", 128),
                    zone.get("reward", 0.0),
                )
            )
        else:
            converted.append(zone)

    return converted