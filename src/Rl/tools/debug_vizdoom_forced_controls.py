#!/usr/bin/env python3
import os
import time
import vizdoom as vzd

def add(game, buttons, name):
    if hasattr(vzd.Button, name):
        game.add_available_button(getattr(vzd.Button, name))
        buttons.append(name)
        return True
    print(f"[missing button] {name}")
    return False

def add_var(game, name):
    if hasattr(vzd.GameVariable, name):
        game.add_available_game_variable(getattr(vzd.GameVariable, name))

game = vzd.DoomGame()
game.set_doom_game_path(os.environ.get("VIZDOOM_IWAD", "/usr/share/games/doom/freedoom1.wad"))
game.set_doom_map(os.environ.get("VIZDOOM_MAP", "E1M1"))
game.set_doom_skill(int(os.environ.get("VIZDOOM_SKILL", "1")))
game.set_window_visible(True)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_episode_timeout(2100)

buttons = []
for name in ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE"]:
    add(game, buttons, name)

for name in ["POSITION_X", "POSITION_Y", "ANGLE", "HEALTH"]:
    add_var(game, name)

print("[buttons]", buttons)

game.init()
game.new_episode()

def vars_now():
    s = game.get_state()
    return None if s is None else list(s.game_variables)

print("[start vars]", vars_now())

# Move forward.
for i in range(90):
    action = [0] * len(buttons)
    if "MOVE_FORWARD" in buttons:
        action[buttons.index("MOVE_FORWARD")] = 1
    game.make_action(action)
    time.sleep(1 / 35)

print("[after forward vars]", vars_now())

# Turn right.
for i in range(90):
    action = [0] * len(buttons)
    if "TURN_RIGHT" in buttons:
        action[buttons.index("TURN_RIGHT")] = 1
    game.make_action(action)
    time.sleep(1 / 35)

print("[after turn vars]", vars_now())

game.close()
print("[debug] done")
