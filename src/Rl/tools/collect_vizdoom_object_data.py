from pathlib import Path
import random
import cv2

from vizdoom import (
    DoomGame,
    Mode,
    ScreenResolution,
    ScreenFormat,
    Button,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vision_dataset_v2" / "objects"

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


OBJECT_CLASSES = [
    "enemy_visible",
    "pickup_health",
    "pickup_ammo",
    "pickup_armor",
    "weapon_pickup",
    "key_pickup",
    "explosive_barrel",
    "projectile_or_attack",
    "no_important_object",
]


MAX_PER_CLASS = {
    "enemy_visible": 1200,
    "pickup_health": 1200,
    "pickup_ammo": 1200,
    "pickup_armor": 1200,
    "weapon_pickup": 800,
    "key_pickup": 800,
    "explosive_barrel": 800,
    "projectile_or_attack": 800,
    "no_important_object": 1200,
}


def setup_folders():
    for class_name in OBJECT_CLASSES:
        (OUT / class_name).mkdir(parents=True, exist_ok=True)


def classify_object(name, category):
    name = str(name).lower().replace("_", " ").replace("-", " ")
    category = str(category).lower().replace("_", " ").replace("-", " ")
    text = f"{name} {category}"

    ignore_words = [
        "player",
        "self",
        "marine",
        "doomplayer",
        "camera",
    ]

    if any(word in text for word in ignore_words):
        return None

    if "monster" in category or "enemy" in category:
        return "enemy_visible"

    if "ammo" in category:
        return "pickup_ammo"

    if "armor" in category or "armour" in category:
        return "pickup_armor"

    if "health" in category:
        return "pickup_health"

    if "key" in category:
        return "key_pickup"

    if "weapon" in category:
        return "weapon_pickup"

    monster_words = [
        "zombie", "zombieman", "shotgun guy", "shotgunguy",
        "chaingunner", "heavy weapon", "imp", "demon", "spectre",
        "cacodemon", "lost soul", "hell knight", "baron",
        "revenant", "mancubus", "arachnotron", "pain elemental",
        "archvile", "arch vile", "cyberdemon", "spider mastermind",

        # Freedoom-style names
        "shotgun zombie", "minigun zombie", "serpentipede",
        "flesh worm", "stealth worm", "trilobite", "matribite",
        "pain bringer", "pain lord", "combat slug", "octaminator",
        "hatchling", "dark soldier", "summoner", "flame bringer",
        "nukeptile", "lavamander",
    ]

    if any(word in text for word in monster_words):
        return "enemy_visible"

    health_words = [
        "health", "stimpack", "stimpak", "medikit", "medkit",
        "soul sphere", "soulsphere", "megahealth", "berserk",
        "health bonus",
    ]

    if any(word in text for word in health_words):
        return "pickup_health"

    ammo_words = [
        "ammo", "clip", "bullets", "bullet", "shell", "shells",
        "rocket", "rockets", "cell", "cells", "backpack",
        "box of bullets", "box of shells", "rocket box", "cell pack",
    ]

    if any(word in text for word in ammo_words):
        return "pickup_ammo"

    armor_words = [
        "armor", "armour", "green armor", "blue armor",
        "security armor", "combat armor", "armor bonus", "megaarmor",
    ]

    if any(word in text for word in armor_words):
        return "pickup_armor"

    weapon_words = [
        "shotgun", "super shotgun", "chaingun", "rocket launcher",
        "plasma", "plasma rifle", "bfg", "bfg9000", "chainsaw",
    ]

    if any(word in text for word in weapon_words):
        return "weapon_pickup"

    key_words = [
        "key", "keycard", "card", "skull",
        "blue key", "red key", "yellow key",
        "blue card", "red card", "yellow card",
        "blue skull", "red skull", "yellow skull",
    ]

    if any(word in text for word in key_words):
        return "key_pickup"

    barrel_words = [
        "barrel", "explosive barrel", "nukage barrel", "burning barrel",
    ]

    if any(word in text for word in barrel_words):
        return "explosive_barrel"

    projectile_words = [
        "projectile",
        "fireball",
        "missile",
        "plasma ball",
        "bfg ball",
        "tracer",
        "doomimpball",
        "impball",
        "imp ball",
        "cacodemonball",
        "baronball",
    ]

    if any(word in text for word in projectile_words):
        return "projectile_or_attack"

    barrel_words = [
        "barrel",
        "explosive barrel",
        "nukage barrel",
        "burning barrel",
    ]

    if any(word in text for word in barrel_words):
        return "explosive_barrel"


def setup_game():
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
    game.set_render_weapon(False)

    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)

    # Buttons must be added before init.
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.MOVE_BACKWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_button(Button.USE)

    game.init()
    return game


def choose_action():
    # Button order:
    # MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT,
    # MOVE_LEFT, MOVE_RIGHT, ATTACK, USE

    actions = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # forward
        [1, 0, 0, 1, 0, 0, 0, 0],  # forward + turn right
        [1, 0, 1, 0, 0, 0, 0, 0],  # forward + turn left
        [0, 0, 1, 0, 0, 0, 0, 0],  # turn left
        [0, 0, 0, 1, 0, 0, 0, 0],  # turn right
        [0, 0, 0, 0, 1, 0, 0, 0],  # strafe left
        [0, 0, 0, 0, 0, 1, 0, 0],  # strafe right
        [0, 1, 0, 0, 0, 0, 0, 0],  # backward
        [0, 0, 0, 0, 0, 0, 1, 0],  # attack
        [0, 0, 0, 0, 0, 0, 0, 1],  # use
    ]

    weights = [
        0.35,
        0.15,
        0.15,
        0.08,
        0.08,
        0.05,
        0.05,
        0.03,
        0.04,
        0.02,
    ]

    return random.choices(actions, weights=weights, k=1)[0]


def main():
    setup_folders()

    game = setup_game()
    game.new_episode()

    saved_counts = {class_name: 0 for class_name in OBJECT_CLASSES}
    seen_debug_names = set()

    max_frames = 12000
    frame_id = 0

    while frame_id < max_frames:
        if game.is_episode_finished():
            game.new_episode()

        state = game.get_state()

        if state is not None and state.screen_buffer is not None:
            frame_rgb = state.screen_buffer
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            labels_found = set()

            if state.labels is not None:
                for obj in state.labels:
                    obj_name = getattr(obj, "object_name", "")
                    obj_category = getattr(obj, "object_category", "")

                    debug_key = (str(obj_name), str(obj_category))
                    if debug_key not in seen_debug_names and len(seen_debug_names) < 80:
                        seen_debug_names.add(debug_key)
                        print(f"[label_debug] name={obj_name} category={obj_category}")

                    class_name = classify_object(obj_name, obj_category)

                    if class_name is not None:
                        labels_found.add(class_name)

            if not labels_found:
                labels_found.add("no_important_object")

            # Save every 5th frame only to reduce near-duplicates.
            if frame_id % 5 == 0:
                for class_name in labels_found:
                    if saved_counts[class_name] >= MAX_PER_CLASS[class_name]:
                        continue

                    # Do not flood no-object data.
                    if class_name == "no_important_object" and random.random() > 0.25:
                        continue

                    out_path = OUT / class_name / f"vizdoom_{frame_id:06d}.jpg"
                    cv2.imwrite(str(out_path), frame_bgr)
                    saved_counts[class_name] += 1

        action = choose_action()
        game.make_action(action, 4)

        frame_id += 1

    game.close()

    print("\nSaved object-labeled ViZDoom frames:")
    for class_name, count in saved_counts.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()