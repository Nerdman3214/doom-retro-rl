from pathlib import Path
import cv2
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vision_dataset_v2" / "autolabeled_vizdoom"

# Use Freedoom if available.
POSSIBLE_WADS = [
    "/usr/share/games/doom/freedoom1.wad",
    "/usr/share/games/doom/freedoom2.wad",
]

WAD_PATH = None
for p in POSSIBLE_WADS:
    if Path(p).exists():
        WAD_PATH = p
        break

if WAD_PATH is None:
    raise FileNotFoundError("Could not find freedoom1.wad or freedoom2.wad")


def make_dirs():
    for name in [
        "enemy_visible",
        "pickup_health",
        "pickup_ammo",
        "pickup_armor",
        "weapon_pickup",
        "key_pickup",
        "explosive_barrel",
        "no_important_object",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def classify_object_name(name, category):
    text = f"{name} {category}".lower()

    if "monster" in text or "zombie" in text or "imp" in text or "demon" in text:
        return "enemy_visible"

    if "health" in text or "medikit" in text or "stimpack" in text:
        return "pickup_health"

    if "ammo" in text or "clip" in text or "shell" in text or "rocket" in text or "cell" in text:
        return "pickup_ammo"

    if "armor" in text:
        return "pickup_armor"

    if "weapon" in text or "shotgun" in text or "chaingun" in text or "launcher" in text:
        return "weapon_pickup"

    if "key" in text or "card" in text or "skull" in text:
        return "key_pickup"

    if "barrel" in text:
        return "explosive_barrel"

    return None


def main():
    make_dirs()

    game = DoomGame()
    game.set_doom_game_path(WAD_PATH)

    # Freedoom Phase 1 usually has ExMx maps.
    # If this fails, change to "map01" for freedoom2.wad.
    game.set_doom_map("e1m1")

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

    frame_id = 0

    while not game.is_episode_finished() and frame_id < 3000:
        state = game.get_state()

        if state is not None and state.screen_buffer is not None:
            frame = state.screen_buffer
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            labels = set()

            if state.labels is not None:
                for obj in state.labels:
                    label = classify_object_name(
                        getattr(obj, "object_name", ""),
                        getattr(obj, "object_category", ""),
                    )
                    if label:
                        labels.add(label)

            if not labels:
                labels.add("no_important_object")

            for label in labels:
                out_path = OUT / label / f"vizdoom_{frame_id:06d}.jpg"
                cv2.imwrite(str(out_path), frame_bgr)

        # Simple exploration actions.
        # [attack, use, jump, crouch, turn_left, turn_right, move_left, move_right, move_forward, move_backward]
        # Default ViZDoom config may not expose these buttons unless scenario config defines them,
        # so use make_action([]) as a safe basic tick.
        game.make_action([], 1)

        frame_id += 1

    game.close()
    print(f"Saved frames to: {OUT}")


if __name__ == "__main__":
    main()