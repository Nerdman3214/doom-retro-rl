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

            window_ids = result.decode().splitlines()

            if not window_ids:
                return None

            # Use the last matching window.
            # This is often the newest Doom window.
            return window_ids[-1]

        except Exception:
            return None

    def _focus(self):
        if self.window_id is None:
            self.window_id = self._find_window()

        if self.window_id is None:
            print("[controller] No Doom window found.")
            return False

        try:
            subprocess.call(
                ["xdotool", "windowactivate", "--sync", str(self.window_id)],
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.03)

            subprocess.call(
                ["xdotool", "windowfocus", "--sync", str(self.window_id)],
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.03)

            active = subprocess.check_output(
                ["xdotool", "getactivewindow"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            if active != str(self.window_id):
                print(
                    f"[controller] Focus failed. "
                    f"wanted={self.window_id} active={active}. "
                    f"Refreshing window id..."
                )

                new_id = self._find_window()

                if new_id is not None:
                    self.window_id = new_id
                    subprocess.call(
                        ["xdotool", "windowactivate", "--sync", str(self.window_id)],
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(0.03)

                    active = subprocess.check_output(
                        ["xdotool", "getactivewindow"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()

                    if active == str(self.window_id):
                        return True

                return False

            return True

        except Exception as e:
            print(f"[controller] Focus error: {e}")
            return False
        
    def hold_key(self, key, duration=0.08):
        if not self._focus():
            print(f"[controller] Not sending key={key}; Doom is not focused.")
            return

        subprocess.call(
            ["xdotool", "keydown", "--window", self.window_id, key],
            stderr=subprocess.DEVNULL,
        )

        time.sleep(duration)

        subprocess.call(
            ["xdotool", "keyup", "--window", self.window_id, key],
            stderr=subprocess.DEVNULL,
        )

    def press_key(self, key):
        if not self._focus():
            print(f"[controller] Not pressing key={key}; Doom is not focused.")
            return

        subprocess.call(
            ["xdotool", "key", "--window", self.window_id, key],
            stderr=subprocess.DEVNULL,
        )

    def key_down(self, key):
        if key in self.held_keys:
            return

        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "keydown", "--window", self.window_id, key],
            stderr=subprocess.DEVNULL,
        )

        self.held_keys.add(key)

    def key_up(self, key):
        if key not in self.held_keys:
            return

        if not self._focus():
            return

        subprocess.call(
            ["xdotool", "keyup", "--window", self.window_id, key],
            stderr=subprocess.DEVNULL,
        )

        self.held_keys.remove(key)

    def release_all(self):
        keys = [
            "w",
            "s",
            "a",
            "d",
            "Left",
            "Right",
            "ctrl",
            "Control_L",
            "e",
        ]

        if self.window_id is None:
            self.window_id = self._find_window()

        if self.window_id is not None:
            for key in keys:
                subprocess.call(
                    ["xdotool", "keyup", "--window", self.window_id, key],
                    stderr=subprocess.DEVNULL,
                )

        self.held_keys.clear()

    def start_move_forward(self):
        self.hold_key("w", duration=0.10)

    def start_move_backward(self):
        self.hold_key("s", duration=0.10)

    def start_turn_left(self):
        self.hold_key("Left", duration=0.06)

    def start_turn_right(self):
        self.hold_key("Right", duration=0.06)

    def start_strafe_left(self):
        self.hold_key("a", duration=0.10)

    def start_strafe_right(self):
        self.hold_key("d", duration=0.10)

    def quick_turn_left(self):
        self.hold_key("Left", duration=0.05)

    def quick_turn_right(self):
        self.hold_key("Right", duration=0.05)

    def stop_turn(self):
        self.release_all()

    def stop_forward_backward(self):
        self.release_all()

    def stop_movement(self):
        self.release_all()

    def shoot(self):
        self.press_key("ctrl")

    def use(self):
        self.press_key("e")

    def swap_weapon(self, slot=1):
        self.press_key(str(slot))