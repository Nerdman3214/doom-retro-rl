# doom_controller.py

import subprocess
import time


class DoomController:

    def __init__(self):
        self.window_name = "DOOM Retro"
        self.window_id = self._find_window()

    def _find_window(self):

        try:

            result = subprocess.check_output([
                "xdotool",
                "search",
                "--name",
                self.window_name
            ])

            return result.decode().splitlines()[0]

        except Exception:
            
            return None

    def _xkey(self, *keys):
        """Send one or more keys to the DOOM window using xdotool."""
        if not self.window_id:
            return
        subprocess.call(
            ["xdotool", "windowfocus", "--sync", self.window_id, "key"] + list(keys),
            stderr=subprocess.DEVNULL
        )   

    def move_forward(self):
        self._xkey("w")

    def move_backward(self):
        self._xkey("s")

    def turn_left(self):
        self._xkey("a")

    def turn_right(self):
        self._xkey("d")

    def shoot(self):
        self._xkey("ctrl")

    def use(self):
        self._xkey("e")

    def swap_weapon(self):
        self._xkey("1", "2", "3", "4", "5", "6", "7", "8", "9")