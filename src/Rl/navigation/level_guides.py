DEFAULT_EMPTY_GUIDE = {
    "player_start": None,
    "main_goal": None,
    "checkpoints": [],
    "secrets": [],
    "route_zones": [],
    "safe_bounds": None,
}


LEVEL_GUIDES = {
    "freedoom1_e1m1": {
        "player_start": {"x": -416.0, "y": 256.0, "angle": 0.0},

        "main_goal": {
            "name": "level_exit",
            "x": -400.0,
            "y": 1296.0,
            "radius": 160.0,
        },

        "safe_bounds": {
            "min_x": -900.0,
            "max_x": 1200.0,
            "min_y": -1200.0,
            "max_y": 2200.0,
        },

        # Main route: corridor is sloped, and continuation exits left.
        # Keep radii tight enough so they do not all fire at spawn.
        "route_zones": [
            {
                "name": "spawn_exit_lane",
                "x": -416.0,
                "y": 256.0,
                "radius": 80.0,
                "reward": 0.0,
                "hint": "leave_spawn",
            },
            {
                "name": "sloped_corridor_entry",
                "x": -360.0,
                "y": 330.0,
                "radius": 75.0,
                "reward": 0.0,
                "hint": "follow_corridor_slope",
            },
            {
                "name": "sloped_corridor_mid",
                "x": -285.0,
                "y": 405.0,
                "radius": 80.0,
                "reward": 0.0,
                "hint": "follow_corridor_slope",
            },
            {
                "name": "corridor_left_turn",
                "x": -425.0,
                "y": 430.0,
                "radius": 90.0,
                "reward": 0.0,
                "hint": "turn_left",
            },
            {
                "name": "post_corridor_room",
                "x": -430.0,
                "y": 620.0,
                "radius": 150.0,
                "reward": 0.0,
                "hint": "advance_after_left_turn",
            },
            {
                "name": "exit_approach",
                "x": -400.0,
                "y": 1000.0,
                "radius": 190.0,
                "reward": 0.0,
                "hint": "advance_to_exit",
            },
            {
                "name": "level_exit",
                "x": -400.0,
                "y": 1296.0,
                "radius": 160.0,
                "reward": 100.0,
                "hint": "finish_level",
            },
        ],

        "checkpoints": [],

        # Optional 100% objectives. Do not make these mandatory yet.
        "secrets": [
            {"name": "secret_sector_52", "x": 496.0, "y": 712.0, "radius": 160.0},
            {"name": "secret_sector_86", "x": 447.0, "y": 1862.0, "radius": 160.0},
            {"name": "secret_sector_128", "x": 1680.0, "y": -880.0, "radius": 180.0},
            {"name": "secret_sector_132", "x": 544.0, "y": 804.0, "radius": 160.0},
        ],
    }
}


def get_level_guide(level_name):
    return LEVEL_GUIDES.get(level_name, DEFAULT_EMPTY_GUIDE)


# ---------------------------------------------------------
# Backward-compatibility helpers
# ---------------------------------------------------------
# Older project files such as env/shared_doom_logic.py expect these functions.
# Keep them so new mission-plan logic and old shared logic can both work.

def get_route_zones(level_name="freedoom1_e1m1"):
    guide = get_level_guide(level_name)
    return guide.get("route_zones", [])


def get_checkpoints(level_name="freedoom1_e1m1"):
    guide = get_level_guide(level_name)
    return guide.get("checkpoints", [])


def get_secrets(level_name="freedoom1_e1m1"):
    guide = get_level_guide(level_name)
    return guide.get("secrets", [])


def get_main_goal(level_name="freedoom1_e1m1"):
    guide = get_level_guide(level_name)
    return guide.get("main_goal")


def get_player_start(level_name="freedoom1_e1m1"):
    guide = get_level_guide(level_name)
    return guide.get("player_start")
