"""
shm_reader.py — Read live player state from Doom Retro via shared memory.

Doom Retro must be compiled with the RL_WriteState() hook in g_game.c,
which writes player data to /dev/shm/doomretro_rl every game tic.

Struct layout (little-endian, all 4-byte integers):
  int32  health
  int32  ammo[0]   (bullets)
  int32  ammo[1]   (shells)
  int32  ammo[2]   (cells)
  int32  ammo[3]   (rockets)
  int32  x         (fixed_t: divide by 65536 to get map units)
  int32  y         (fixed_t: divide by 65536 to get map units)
  uint32 angle     (angle_t: 0xFFFFFFFF = 360°)
  int32  armor
  int32  kills
"""

import struct
import os

_SHM_PATH = "/dev/shm/doomretro_rl"

# Format string for struct.unpack: 9 ints = health + ammo[4] + x + y + angle + armor + kills
# angle is uint32, rest are int32
_FMT = "=iiiiiiiIii"
_SIZE = struct.calcsize(_FMT)


def is_available() -> bool:
    """Return True if the shared memory segment exists (Doom is running with the hook)."""
    return os.path.exists(_SHM_PATH)


def read_player_state() -> dict | None:
    """
    Read the current player state from shared memory.

    Returns a dict, or None if shared memory is not available yet.

    Keys:
        health  : int   (0-200)
        ammo    : list[int]  [bullets, shells, cells, rockets]
        x       : float  map X position in map units
        y       : float  map Y position in map units
        angle   : float  facing angle in degrees (0-360)
        armor   : int
        kills   : int
    """
    if not os.path.exists(_SHM_PATH):
        return None
    try:
        with open(_SHM_PATH, "rb") as f:
            data = f.read(_SIZE)
        if len(data) < _SIZE:
            return None
        vals = struct.unpack(_FMT, data)
        health, b, sh, ce, ro, x, y, angle, armor, kills = vals
        return {
            "health": health,
            "ammo":   [b, sh, ce, ro],
            "x":      x / 65536.0,
            "y":      y / 65536.0,
            "angle":  angle / 4294967296.0 * 360.0,
            "armor":  armor,
            "kills":  kills,
        }
    except OSError:
        return None


if __name__ == "__main__":
    # Quick test — run while Doom Retro is open
    import time
    print(f"Shared memory path: {_SHM_PATH}")
    print(f"Struct size: {_SIZE} bytes")
    if not is_available():
        print("Shared memory not found. Is Doom Retro running with the RL hook compiled in?")
    else:
        for _ in range(5):
            state = read_player_state()
            print(state)
            time.sleep(0.5)
