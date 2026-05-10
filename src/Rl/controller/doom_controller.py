
import subprocess
import time


class DoomController:
    class RotationState:
        def __init__(self):
            self.turn_velocity = 0.0

        def update(self, action_turn_signal, decay=0.4):
            self.turn_velocity = (
                self.turn_velocity * decay + action_turn_signal
            )
            return self.turn_velocity

    def __init__(self):
        self.window_name = "DOOM Retro"
        self.window_id = self._find_window()
        self.held_keys = set()

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
        if not self.window_id:
            return

        subprocess.call([
            "xdotool",
            "windowactivate",
            "--sync",
            self.window_id
        ])

        subprocess.call([
            "xdotool",
            "key"
        ] + list(keys), stderr=subprocess.DEVNULL)

    def _focus(self):
        if not self.window_id:
            return

        subprocess.call([
            "xdotool",
            "windowactivate",
            "--sync",
            self.window_id
        ])

    def _tap(self, key):
        """Simulate a quick key press and release."""
        self._xkey(key)
        time.sleep(0.1)  # Short press duration
        self._xkey(key)

    def key_down(self, key):
        if key in self.held_keys:
            return

        self._focus()
        subprocess.call(["xdotool", "keydown", key])
        self.held_keys.add(key)

    def key_up(self, key):
        if key not in self.held_keys:
            return

        self._focus()
        subprocess.call(["xdotool", "keyup", key])
        self.held_keys.remove(key)

    def hold_key(self, key, duration=0.2):
        if not self.window_id:
            return
        subprocess.call(["xdotool", "keydown", key])
        time.sleep(duration)
        subprocess.call(["xdotool", "keyup", key])

    def start_move_forward(self):
        self.key_down("w")

    def start_move_backward(self):
        self.key_down("s")

    def stop_forward_backward(self):
        self.key_up("w")
        self.key_up("s")
    
    def move_forward(self, hold=True):
        if hold:
            self.key_down("w")
        else:
            self._tap("w")

    def start_turn_left(self):
        self.key_down("Left")

    def start_turn_right(self):
        self.key_down("Right")

    def stop_turn(self):
        self.key_up("Left")
        self.key_up("Right")

    def strafe_left(self):
        self.key_down("a")

    def strafe_right(self):
        self.key_down("d")

    def move_backward(self):
        self.hold_key("s")

    def shoot(self):
        self._xkey("ctrl")

    def use(self):
        self._xkey("e")

    def swap_weapon(self, slot=1):
        self._xkey(str(slot))

    def stop_movement(self):
        self.key_up("w")
        self.key_up("s")
