"""
Map-aware level guides for Doom/Freedoom RL.

These coordinates come from the actual Freedoom E1M1 WAD:
- Player start: (-416, 256)
- Real exit line midpoint: (-400, 1296)

Rule:
Do not add fake old route names such as spawn_exit/right_route/combat_corridor.
All route help must be based on real WAD coordinates.
"""

DEFAULT_EMPTY_GUIDE = {
    "name": "generic",
    "player_start": None,
    "main_goal": None,
    "route_zones": [],
    "waypoints": [],
    "safe_bounds": None,
    "secrets": [],
    "keys": [],
    "locked_doors": [],
    "switches": [],
    "use_points": [],
}


LEVEL_GUIDES = {
    "freedoom1_e1m1": {
        "name": "freedoom1_e1m1",

        "player_start": {
            "name": "player_start",
            "x": -416.0,
            "y": 256.0,
            "angle": 0.0,
        },

        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 96.0,
            "reward": 10.0,
        },

        # Real-coordinate soft corridor toward the exit.
        # These are not fake route names. They are broad navigation bubbles.
        "waypoints": [
            {"name": "spawn_exit_lane", "x": -416.0, "y": 384.0, "radius": 128.0, "reward": 0.15},
            {"name": "mid_room_entry", "x": -400.0, "y": 480.0, "radius": 144.0, "reward": 0.25},
            {"name": "mid_room_progress", "x": -360.0, "y": 640.0, "radius": 144.0, "reward": 0.40},
            {"name": "exit_lane_rejoin", "x": -400.0, "y": 850.0, "radius": 160.0, "reward": 0.60},
            {"name": "exit_approach", "x": -400.0, "y": 1100.0, "radius": 160.0, "reward": 0.85},
            {"name": "level_exit", "x": -400.0, "y": 1296.0, "radius": 128.0, "reward": 2.50},
        ],

        # route_zones is kept for compatibility with old helper code,
        # but now it mirrors real waypoints only.
        "route_zones": [
            {"name": "spawn_exit_lane", "x": -416.0, "y": 256.0, "radius": 128.0, "reward": 0.15},

            # Sloped corridor path
            {"name": "sloped_corridor_entry", "x": -360.0, "y": 320.0, "radius": 144.0, "reward": 0.30},
            {"name": "sloped_corridor_mid", "x": -300.0, "y": 390.0, "radius": 144.0, "reward": 0.40},

            # Important: corridor exit is to the left
            {"name": "corridor_left_turn", "x": -420.0, "y": 430.0, "radius": 160.0, "reward": 0.75},

            {"name": "post_corridor_room", "x": -430.0, "y": 620.0, "radius": 180.0, "reward": 1.00},
            {"name": "exit_approach", "x": -400.0, "y": 1000.0, "radius": 180.0, "reward": 1.50},
            {"name": "level_exit", "x": -400.0, "y": 1296.0, "radius": 128.0, "reward": 3.00},
        ],

        # Soft bounds to stop the agent from drifting into the old east/right trap.
        "safe_bounds": {
            "min_x": -900.0,
            "max_x": 200.0,
            "min_y": -128.0,
            "max_y": 1500.0,
            "soft_penalty": -0.02,
            "hard_penalty": -0.08,
        },

        # Real secret centers, but disabled for level-completion training.
        "secrets": [
            {"name": "secret_52", "sector": 52, "x": 496.0, "y": 712.0, "radius": 96.0, "reward": 0.0, "enabled": False},
            {"name": "secret_86", "sector": 86, "x": 447.0, "y": 1862.0, "radius": 96.0, "reward": 0.0, "enabled": False},
            {"name": "secret_128", "sector": 128, "x": 1680.0, "y": -880.0, "radius": 96.0, "reward": 0.0, "enabled": False},
            {"name": "secret_132", "sector": 132, "x": 544.0, "y": 804.0, "radius": 96.0, "reward": 0.0, "enabled": False},
        ],

        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],
    }
}


def normalize_level_name(level_name=None):
    name = str(level_name or "freedoom1_e1m1").lower()
    aliases = {
        "e1m1": "freedoom1_e1m1",
        "doom_e1m1": "freedoom1_e1m1",
        "freedoom_e1m1": "freedoom1_e1m1",
        "freedoom1_e1m1": "freedoom1_e1m1",
    }
    return aliases.get(name, name)


def get_level_guide(level_name=None):
    name = normalize_level_name(level_name)
    guide = LEVEL_GUIDES.get(name)

    if guide is None:
        return DEFAULT_EMPTY_GUIDE.copy()

    merged = DEFAULT_EMPTY_GUIDE.copy()
    merged.update(guide)
    return merged


def get_main_goal(level_name=None):
    return get_level_guide(level_name).get("main_goal")


def get_route_zones(level_name=None):
    return list(get_level_guide(level_name).get("route_zones", []))


def get_waypoints(level_name=None):
    return list(get_level_guide(level_name).get("waypoints", []))


def get_enabled_secrets(level_name=None):
    return [
        s for s in get_level_guide(level_name).get("secrets", [])
        if s.get("enabled", False)
    ]
