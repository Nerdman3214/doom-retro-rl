from pathlib import Path
from collections import Counter

from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat


POSSIBLE_WADS = [
    "/usr/share/games/doom/freedoom1.wad",
    "/usr/share/games/doom/freedoom2.wad",
]

WAD_PATH = None
for path in POSSIBLE_WADS:
    if Path(path).exists():
        WAD_PATH = path
        break

if WAD_PATH is None:
    raise FileNotFoundError("Could not find freedoom1.wad or freedoom2.wad.")


game = DoomGame()
game.set_doom_game_path(WAD_PATH)

if "freedoom1" in WAD_PATH:
    game.set_doom_map("e1m1")
else:
    game.set_doom_map("map01")

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_screen_format(ScreenFormat.RGB24)

game.set_labels_buffer_enabled(True)
game.set_objects_info_enabled(True)

game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)

game.set_window_visible(False)
game.set_mode(Mode.PLAYER)

game.init()
game.new_episode()

name_counter = Counter()
category_counter = Counter()

for frame_id in range(100):
    state = game.get_state()

    if state is not None and state.labels is not None:
        print(f"\n--- FRAME {frame_id} ---")

        for obj in state.labels:
            name = getattr(obj, "object_name", "")
            category = getattr(obj, "object_category", "")

            name_counter[name] += 1
            category_counter[str(category)] += 1

            print(
                "name=", name,
                "| category=", category,
                "| x=", getattr(obj, "x", None),
                "| y=", getattr(obj, "y", None),
                "| w=", getattr(obj, "width", None),
                "| h=", getattr(obj, "height", None),
            )

    game.make_action([], 1)

game.close()

print("\nMost common object names:")
for name, count in name_counter.most_common(30):
    print(count, name)

print("\nMost common categories:")
for category, count in category_counter.most_common(30):
    print(count, category)