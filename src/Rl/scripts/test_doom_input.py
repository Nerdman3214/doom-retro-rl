import sys
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from controller.doom_controller import DoomController


def main():
    controller = DoomController()

    print("Window ID:", controller.window_id)

    if controller.window_id is None:
        print("ERROR: Could not find DOOM Retro window.")
        return

    print("Testing movement. Click the Doom window if it does not move.")
    time.sleep(2)

    print("Move forward")
    controller.hold_key("w", duration=1.0)
    time.sleep(0.5)

    print("Move backward")
    controller.hold_key("s", duration=1.0)
    time.sleep(0.5)

    print("Turn right")
    controller.hold_key("Right", duration=0.5)
    time.sleep(0.5)

    print("Turn left")
    controller.hold_key("Left", duration=0.5)
    time.sleep(0.5)

    print("Strafe right")
    controller.hold_key("d", duration=1.0)
    time.sleep(0.5)

    print("Strafe left")
    controller.hold_key("a", duration=1.0)

    controller.release_all()
    print("Done.")


if __name__ == "__main__":
    main()