"""
WAD-based level guide database.

This file stores static map facts only:
- player start
- true main exit
- optional secrets
- weak route zones

RouteDirector decides what target is active.
The env calculates reward from the director output.
"""

DEFAULT_EMPTY_GUIDE = {
    "name": "generic",
    "player_start": None,
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

        # Found from the uploaded freedoom1.wad.
        "player_start": {
            "name": "player_start",
            "x": -416,
            "y": 256,
            "angle": 0,
        },

        # Found from the uploaded freedoom1.wad exit linedef.
        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": -400,
            "y": 1296,
            "radius": 96,
            "reward": 10.0,
        },

        # Weak A -> B hints. These are not prizes; they are breadcrumb markers.
        # If the agent reaches only spawn_forward repeatedly, add/debug the next
        # route point from real trajectory logs before raising these rewards.
        "route_nodes": [
            {
                "name": "spawn_forward",
                "type": "route",
                "x": -416,
                "y": 448,
                "radius": 112,
                "next": ["mid_route_1"],
                "reward": 0.05,
            },
            {
                "name": "mid_route_1",
                "type": "route",
                "x": -416,
                "y": 704,
                "radius": 112,
                "next": ["mid_route_2"],
                "reward": 0.08,
            },
            {
                "name": "mid_route_2",
                "type": "route",
                "x": -416,
                "y": 960,
                "radius": 112,
                "next": ["exit_approach"],
                "reward": 0.10,
            },
            {
                "name": "exit_approach",
                "type": "exit_path",
                "x": -400,
                "y": 1184,
                "radius": 128,
                "next": ["level_exit"],
                "reward": 0.15,
            },
        ],

        "route_zones": [
            {"name": "spawn_forward", "x": -416, "y": 448, "radius": 112, "reward": 0.05},
            {"name": "mid_route_1", "x": -416, "y": 704, "radius": 112, "reward": 0.08},
            {"name": "mid_route_2", "x": -416, "y": 960, "radius": 112, "reward": 0.10},
            {"name": "exit_approach", "x": -400, "y": 1184, "radius": 128, "reward": 0.15},
        ],

        # Secrets found from the WAD, but disabled for basic completion training.
        "secrets": [
            {"name": "secret_52", "sector": 52, "x": 496, "y": 712, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_86", "sector": 86, "x": 447, "y": 1862, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_128", "sector": 128, "x": 1680, "y": -880, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_132", "sector": 132, "x": 544, "y": 804, "radius": 96, "reward": 0.0, "enabled": False},
        ],

        "checkpoints": [],
        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],
        "strategies": [
            "safe_main_route",
            "recover_if_stuck",
            "combat_clear_then_exit",
            "secrets_disabled_until_completion",
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

    Some shared helpers expect route zones as:
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


def get_main_goal(level_name=None):
    guide = get_level_guide(level_name)
    return guide.get("main_goal")


def get_enabled_secrets(level_name=None):
    guide = get_level_guide(level_name)
    return [
        secret
        for secret in guide.get("secrets", [])
        if secret.get("enabled", False)
    ]
