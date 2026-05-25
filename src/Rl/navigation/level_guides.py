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
    "main_goal": None,
    "checkpoints": [],
    "secrets": [],
    "keys": [],
    "locked_doors": [],
    "switches": [],
    "use_points": [],
}


LEVEL_GUIDES = {
    "freedoom1_e1m1": {
        "route_zones": [
            {
                "name": "spawn_exit",
                "x": 600,
                "y": 360,
                "radius": 120,
                "reward": 0.50,
            },
            {
                "name": "right_route",
                "x": 625,
                "y": 360,
                "radius": 120,
                "reward": 0.80,
            },
            {
                "name": "door_area",
                "x": 700,
                "y": 420,
                "radius": 140,
                "reward": 1.00,
            },
            {
                "name": "combat_corridor",
                "x": 850,
                "y": 400,
                "radius": 160,
                "reward": 1.50,
            },
            {
                "name": "exit_route",
                "x": 1000,
                "y": 500,
                "radius": 180,
                "reward": 2.00,
            },
        ],
        "checkpoints": [],
        "secrets": [],
    }
}

"""LEVEL_GUIDES = {
    "freedoom1_e1m1": {
        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": 1200.0,
            "y": -100.0,
            "radius": 160.0,
            "reward": 10.0,
        },

        "checkpoints": [
            {
                "name": "start_exit_path",
                "type": "route",
                "x": 300.0,
                "y": 200.0,
                "radius": 128.0,
                "reward": 3.0,
            },
        ],

        "secrets": [
            {
                "name": "secret_1",
                "type": "secret",
                "x": -208.0,
                "y": 144.0,
                "radius": 128.0,
                "reward": 6.0,
            },
            {
                "name": "secret_2",
                "type": "secret",
                "x": 400.0,
                "y": 300.0,
                "radius": 128.0,
                "reward": 6.0,
            },
        ],

        "keys": [
            # Fill these with real coordinates when you find keycards.
            # Example:
            # {
            #     "name": "blue_keycard",
            #     "type": "key",
            #     "key": "blue",
            #     "x": 500.0,
            #     "y": 300.0,
            #     "radius": 96.0,
            #     "reward": 8.0,
            # },
        ],

        "locked_doors": [
            # Example:
            # {
            #     "name": "blue_exit_door",
            #     "type": "locked_door",
            #     "requires_key": "blue",
            #     "x": 900.0,
            #     "y": 100.0,
            #     "radius": 128.0,
            #     "leads_to": "level_exit",
            # },
        ],

        "switches": [
            # Example:
            # {
            #     "name": "bridge_switch",
            #     "type": "switch",
            #     "x": 700.0,
            #     "y": 250.0,
            #     "radius": 96.0,
            #     "unlocks": "bridge_door",
            #     "reward": 5.0,
            # },
        ],

        "use_points": [
            # Doors/elevators/buttons that should teach the agent to press use.
            # Replace placeholders with real x/y values from logs.
            {
                "name": "possible_first_door_or_elevator",
                "type": "use_point",
                "x": 300.0,
                "y": 200.0,
                "radius": 128.0,
                "reward": 3.0,
            },
        ],
    },

    "freedoom1_e1m2": {
        "main_goal": {
            "name": "level_exit",
            "type": "exit",
            "x": 0.0,
            "y": 0.0,
            "radius": 160.0,
            "reward": 10.0,
        },
        "checkpoints": [],
        "secrets": [],
        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],
    },

    "freedoom2_default": {
        "main_goal": {
            "name": "exit_area",
            "type": "exit",
            "x": 1024.0,
            "y": 512.0,
            "radius": 192.0,
            "reward": 10.0,
        },

        "checkpoints": [
            {
                "name": "leave_start_area",
                "type": "route",
                "x": 0.0,
                "y": 0.0,
                "radius": 128.0,
                "reward": 4.0,
            },
            {
                "name": "first_combat_area",
                "type": "route",
                "x": 256.0,
                "y": 128.0,
                "radius": 128.0,
                "reward": 5.0,
            },
            {
                "name": "first_door_or_opening",
                "type": "route",
                "x": 512.0,
                "y": 256.0,
                "radius": 128.0,
                "reward": 6.0,
            },
            {
                "name": "main_route_midpoint",
                "type": "route",
                "x": 768.0,
                "y": 384.0,
                "radius": 160.0,
                "reward": 8.0,
            },
        ],

        "secrets": [
            {
                "name": "secret_1_placeholder",
                "type": "secret",
                "x": 384.0,
                "y": -256.0,
                "radius": 96.0,
                "reward": 6.0,
            },
        ],

        "keys": [],
        "locked_doors": [],
        "switches": [],
        "use_points": [],
    },
}
"""

def get_level_guide(level_name):
    guide = LEVEL_GUIDES.get(level_name)

    if guide is None:
        return DEFAULT_EMPTY_GUIDE.copy()

    fixed = DEFAULT_EMPTY_GUIDE.copy()
    fixed.update(guide)
    return fixed