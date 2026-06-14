"""
Real WAD-based level guide database.

Current Freedoom E1M1 facts:
- Player start: (-416, 256)
- Real exit:    (-400, 1296)

Important:
- route_zones are intentionally empty right now.
- The old positive-X route was wrong and caused the agent to keep walking east/right.
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

        "player_start": {
            "name": "player_start",
            "x": -416,
            "y": 256,
            "angle": 0,
        },

        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": -400,
            "y": 1296,
            "radius": 96,
            "reward": 10.0,
        },

        # Disabled until the real route is mapped.
        "route_nodes": [],
        "route_zones": [],
        "checkpoints": [],

        # Real secret sector centers, but disabled for now.
        "secrets": [
            {"name": "secret_52", "sector": 52, "x": 496, "y": 712, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_86", "sector": 86, "x": 447, "y": 1862, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_128", "sector": 128, "x": 1680, "y": -880, "radius": 96, "reward": 0.0, "enabled": False},
            {"name": "secret_132", "sector": 132, "x": 544, "y": 804, "radius": 96, "reward": 0.0, "enabled": False},
        ],

        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],

        "strategies": [
            "real_exit_only",
            "secrets_disabled_until_completion",
        ],
    }
}


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
    guide = get_level_guide(level_name)
    return list(guide.get("route_zones", []))


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
