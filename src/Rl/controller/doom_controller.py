import subprocess
import time


class DoomController:
    def __init__(self):
        self.window_name = "DOOM Retro"
        self.window_id = self._find_window()
        self.held_keys = set()

    def _find_window(self):
        try:
            result = subprocess.check_output(
                ["xdotool", "search", "--name", self.window_name],
                stderr=subprocess.DEVNULL,
            )
            ids = result.decode().splitlines()
            return ids[0] if ids else None
        except Exception:
            return None

    def _focus(self):
        """
        Focus the Doom Retro window before sending keys.
        """
        if self.window_id is None:
            self.window_id = self._find_window()

        if self.window_id is None:
            return False

        subprocess.call(
            ["xdotool", "windowactivate", "--sync", self.window_id],
            stderr=subprocess.DEVNULL,
        )
        return True

    def _xkey(self, *keys):
        """
        Send a normal tap key event to the Doom window.
        """
        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "key"] + list(keys),
            stderr=subprocess.DEVNULL,
        )

    def key_down(self, key):
        if key in self.held_keys:
            return

        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "keydown", key],
            stderr=subprocess.DEVNULL,
        )
        self.held_keys.add(key)

    def key_up(self, key):
        if key not in self.held_keys:
            return

        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "keyup", key],
            stderr=subprocess.DEVNULL,
        )
        self.held_keys.remove(key)

    def hold_key(self, key, duration=0.08):
        if not self._focus():
            return

        subprocess.call(["xdotool", "keydown", key], stderr=subprocess.DEVNULL)
        time.sleep(duration)
        subprocess.call(["xdotool", "keyup", key], stderr=subprocess.DEVNULL)

    def release_all(self):
        for key in list(self.held_keys):
            self.key_up(key)

        # Extra safety release in case a key got stuck outside held_keys.
        if self._focus():
            subprocess.call(
                [
                    "xdotool",
                    "keyup",
                    "w",
                    "keyup",
                    "s",
                    "keyup",
                    "a",
                    "keyup",
                    "d",
                    "keyup",
                    "Left",
                    "keyup",
                    "Right",
                    "keyup",
                    "ctrl",
                ],
                stderr=subprocess.DEVNULL,
            )

        self.held_keys.clear()

    # ---------------------------------------------------------
    # Movement
    # ---------------------------------------------------------

    def start_move_forward(self):
        self.key_down("w")

    def start_move_backward(self):
        self.key_down("s")

    def start_turn_left(self):
        self.key_down("Left")

    def start_turn_right(self):
        self.key_down("Right")

    def start_strafe_left(self):
        self.key_down("a")

    def start_strafe_right(self):
        self.key_down("d")

    def quick_turn_left(self):
        self.hold_key("Left", duration=0.08)

    def quick_turn_right(self):
        self.hold_key("Right", duration=0.08)

    def stop_turn(self):
        self.key_up("Left")
        self.key_up("Right")

    def stop_forward_backward(self):
        self.key_up("w")
        self.key_up("s")

    def stop_movement(self):
        self.release_all()

    # ---------------------------------------------------------
    # Combat / interaction
    # ---------------------------------------------------------

    def shoot(self):
        self._xkey("ctrl")

    def use(self):
        self._xkey("e")

    def swap_weapon(self, slot=1):
        self._xkey(str(slot))