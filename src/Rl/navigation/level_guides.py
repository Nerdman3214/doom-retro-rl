"""
Manual checkpoint and secret guides.

Important:
These coordinates are placeholders.

Run your game once with debug x/y printing, write down useful positions,
then replace the placeholder x/y values below.

The tracker will only work well when ObservationBuilder/shared memory
provides real player x/y coordinates.
"""


LEVEL_GUIDES = {
    # Freedoom Phase 1 / default early map guide.
    # Replace these with real coordinates from your run.
    "freedoom1_default": {
        "checkpoints": [
            {
                "name": "leave_start_area",
                "x": 0.0,
                "y": 0.0,
                "radius": 128.0,
                "reward": 4.0,
            },
            {
                "name": "first_pickup_area",
                "x": 256.0,
                "y": 0.0,
                "radius": 128.0,
                "reward": 5.0,
            },
            {
                "name": "first_door_or_switch",
                "x": 512.0,
                "y": 128.0,
                "radius": 128.0,
                "reward": 6.0,
            },
            {
                "name": "mid_level_progress",
                "x": 768.0,
                "y": 256.0,
                "radius": 160.0,
                "reward": 8.0,
            },
            {
                "name": "exit_area",
                "x": 1024.0,
                "y": 512.0,
                "radius": 192.0,
                "reward": 15.0,
            },
        ],
        "secrets": [
            {
                "name": "secret_1_placeholder",
                "x": 300.0,
                "y": -200.0,
                "radius": 96.0,
                "reward": 12.0,
            },
        ],
    },

    # Freedoom Phase 2 / Doom II style default guide.
    # Replace these with real coordinates from your run.
    "freedoom2_default": {
        "checkpoints": [
            {
                "name": "leave_start_area",
                "x": 0.0,
                "y": 0.0,
                "radius": 128.0,
                "reward": 4.0,
            },
            {
                "name": "first_combat_area",
                "x": 256.0,
                "y": 128.0,
                "radius": 128.0,
                "reward": 5.0,
            },
            {
                "name": "first_door_or_opening",
                "x": 512.0,
                "y": 256.0,
                "radius": 128.0,
                "reward": 6.0,
            },
            {
                "name": "main_route_midpoint",
                "x": 768.0,
                "y": 384.0,
                "radius": 160.0,
                "reward": 8.0,
            },
            {
                "name": "exit_area",
                "x": 1024.0,
                "y": 512.0,
                "radius": 192.0,
                "reward": 15.0,
            },
        ],
        "secrets": [
            {
                "name": "secret_1_placeholder",
                "x": 384.0,
                "y": -256.0,
                "radius": 96.0,
                "reward": 12.0,
            },
        ],
    },
}


def get_level_guide(name="freedoom1_default"):
    return LEVEL_GUIDES.get(name, LEVEL_GUIDES["freedoom1_default"])