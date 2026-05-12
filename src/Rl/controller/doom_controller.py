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

    def refresh_window(self):
        self.window_id = self._find_window()
        return self.window_id is not None

    def _focus(self):
        if not self.window_id:
            self.refresh_window()

        if not self.window_id:
            return False

        subprocess.call(
            ["xdotool", "windowactivate", "--sync", self.window_id],
            stderr=subprocess.DEVNULL,
        )
        return True

    def _tap(self, key):
        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "key", key],
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

    def release_all(self):
        for key in list(self.held_keys):
            self.key_up(key)

    # -------------------------
    # Movement
    # -------------------------

    def start_move_forward(self):
        self.key_down("w")

    def start_move_backward(self):
        self.key_down("s")

    def start_strafe_left(self):
        self.key_down("a")

    def start_strafe_right(self):
        self.key_down("d")

    def start_turn_left(self):
        self.key_down("Left")

    def start_turn_right(self):
        self.key_down("Right")

    def quick_turn_left(self):
        self.key_down("a")
        time.sleep(0.10)
        self.key_up("a")


    def quick_turn_right(self):
        self.key_down("d")
        time.sleep(0.10)
        self.key_up("d")

    # -------------------------
    # Tap actions
    # -------------------------

    def shoot(self):
        self._tap("ctrl")

    def use(self):
        self._tap("e")

    def swap_weapon(self, slot=1):
        self._tap(str(slot))

    # -------------------------
    # Compatibility aliases
    # -------------------------

    def stop_forward_backward(self):
        self.key_up("w")
        self.key_up("s")

    def stop_turn(self):
        self.key_up("Left")
        self.key_up("Right")

    def stop_movement(self):
        self.release_all()

    def strafe_left(self):
        self.start_strafe_left()

    def strafe_right(self):
        self.start_strafe_right()