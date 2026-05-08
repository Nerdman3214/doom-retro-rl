import subprocess
import time

from utils.window_focus import find_doom_window

# xdotool key names for DOOM Retro defaults
_XDOTOOL_KEYS = {
    "up":    "Up",
    "down":  "Down",
    "left":  "Left",
    "right": "Right",
    "ctrl":  "ctrl",
    "space": "space",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
}

USE_COOLDOWN = 1.5  # seconds between "use" presses (prevents door spam)


class DoomController:

    def __init__(self):
        self.window_id = find_doom_window()
        self._last_use_time = 0.0
        if self.window_id:
            print(f"Found DOOM window: {self.window_id}")
        else:
            print("WARNING: DOOM window not found. Is DOOM Retro running?")

    def focus_game(self):
        wid = find_doom_window()
        if wid:
            self.window_id = wid

    def focus_window(self):
        if self.window_id:
            subprocess.call(
                ["xdotool", "windowactivate", "--sync", self.window_id],
                stderr=subprocess.DEVNULL
            )

    def _xkey(self, *keys):
        """Focus DOOM + send keys atomically in a single xdotool call."""
        if not self.window_id:
            return
        xkeys = [_XDOTOOL_KEYS.get(k, k) for k in keys]
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id, "key"] + xkeys,
            stderr=subprocess.DEVNULL
        )

    def _xkey_held(self, key, duration=0.10):
        """Focus DOOM + keydown, sleep, keyup — for held movement keys."""
        if not self.window_id:
            return
        xkey = _XDOTOOL_KEYS.get(key, key)
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id,
             "keydown", xkey],
            stderr=subprocess.DEVNULL
        )
        time.sleep(duration)
        subprocess.call(
            ["xdotool", "keyup", xkey],
            stderr=subprocess.DEVNULL
        )

    def tap(self, key, duration=0.05):
        self._xkey_held(key, duration)

    # --- Movement ---
    def move_forward(self):   self._xkey_held("up",    0.06)
    def move_backward(self):  self._xkey_held("down",  0.06)
    def move_left(self):      self._xkey_held("left",  0.06)
    def move_right(self):     self._xkey_held("right", 0.06)
    def turn_left(self):      self._xkey_held("left",  0.06)
    def turn_right(self):     self._xkey_held("right", 0.06)

    # --- Combat ---
    def shoot(self):          self._xkey("ctrl")

    def use(self):
        now = time.time()
        if now - self._last_use_time < USE_COOLDOWN:
            return
        self._last_use_time = now
        self._xkey("space")

    # --- Combo actions ---
    def move_forward_shoot(self):
        xkey_up   = _XDOTOOL_KEYS["up"]
        xkey_ctrl = _XDOTOOL_KEYS["ctrl"]
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id,
             "keydown", xkey_up, "key", xkey_ctrl, "keyup", xkey_up],
            stderr=subprocess.DEVNULL
        )

    def move_backward_shoot(self):
        xkey_dn   = _XDOTOOL_KEYS["down"]
        xkey_ctrl = _XDOTOOL_KEYS["ctrl"]
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id,
             "keydown", xkey_dn, "key", xkey_ctrl, "keyup", xkey_dn],
            stderr=subprocess.DEVNULL
        )

    def turn_left_shoot(self):
        xkey_l    = _XDOTOOL_KEYS["left"]
        xkey_ctrl = _XDOTOOL_KEYS["ctrl"]
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id,
             "keydown", xkey_l, "key", xkey_ctrl, "keyup", xkey_l],
            stderr=subprocess.DEVNULL
        )

    def turn_right_shoot(self):
        xkey_r    = _XDOTOOL_KEYS["right"]
        xkey_ctrl = _XDOTOOL_KEYS["ctrl"]
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id,
             "keydown", xkey_r, "key", xkey_ctrl, "keyup", xkey_r],
            stderr=subprocess.DEVNULL
        )

    # --- Weapon switching ---
    def swap_weapon(self):
        for k in ["2", "3", "4", "5", "6", "7"]:
            self._xkey(k)

    # --- Camera ---
    def move_camera(self, direction, duration=0.1):
        key_map = {"left": "left", "right": "right",
                   "forward": "up", "backward": "down"}
        self._xkey_held(key_map.get(direction, "up"), duration)