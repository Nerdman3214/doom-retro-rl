from pathlib import Path
import csv
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

OUT_DIR = ROOT / "vision_dataset_v2" / "object_multilabel"
IMAGE_DIR = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "labels.csv"

POSSIBLE_WADS = [
    "/usr/share/games/doom/freedoom1.wad",
    "/usr/share/games/doom/freedoom2.wad",
]

LABEL_COLUMNS = [
    "enemy_visible",
    "pickup_health",
    "pickup_ammo",
    "pickup_armor",
    "explosive_barrel",
    "no_important_object",
]


def find_wad():
    for path in POSSIBLE_WADS:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Could not find freedoom1.wad or freedoom2.wad.")


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
        "dead",
        "gib",
        "gore",
        "blood",
        "meat",
        "corpse",
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

    if "hazard" in category and "barrel" in text:
        return "explosive_barrel"

    monster_words = [
        "zombie",
        "zombieman",
        "shotgun guy",
        "shotgunguy",
        "chaingunner",
        "heavy weapon",
        "imp",
        "doomimp",
        "demon",
        "spectre",
        "cacodemon",
        "lost soul",
        "hell knight",
        "baron",
        "revenant",
        "mancubus",
        "arachnotron",
        "pain elemental",
        "archvile",
        "cyberdemon",
        "spider mastermind",
        "serpentipede",
        "flesh worm",
        "stealth worm",
    ]

    if any(word in text for word in monster_words):
        return "enemy_visible"

    health_words = [
        "health",
        "stimpack",
        "stimpak",
        "medikit",
        "medkit",
        "soul sphere",
        "soulsphere",
        "megahealth",
        "berserk",
        "health bonus",
    ]

    if any(word in text for word in health_words):
        return "pickup_health"

    ammo_words = [
        "ammo",
        "clip",
        "bullets",
        "bullet",
        "shell",
        "shells",
        "rocket",
        "rockets",
        "cell",
        "cells",
        "backpack",
    ]

    if any(word in text for word in ammo_words):
        return "pickup_ammo"

    armor_words = [
        "armor",
        "armour",
        "armor bonus",
        "green armor",
        "blue armor",
        "security armor",
        "combat armor",
        "megaarmor",
    ]

    if any(word in text for word in armor_words):
        return "pickup_armor"

    barrel_words = [
        "explosivebarrel",
        "explosive barrel",
        "barrel",
    ]

    if any(word in text for word in barrel_words):
        return "explosive_barrel"

    return None


def setup_game():
    wad_path = find_wad()

    game = DoomGame()
    game.set_doom_game_path(wad_path)

    if "freedoom1" in wad_path:
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
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    game = setup_game()
    game.new_episode()

    max_frames = 15000
    save_every = 5

    rows = []
    counts = {label: 0 for label in LABEL_COLUMNS}

    for frame_id in range(max_frames):
        if game.is_episode_finished():
            game.new_episode()

        state = game.get_state()

        if state is not None and state.screen_buffer is not None:
            labels = {label: 0 for label in LABEL_COLUMNS}

            if state.labels is not None:
                for obj in state.labels:
                    class_name = classify_object(
                        getattr(obj, "object_name", ""),
                        getattr(obj, "object_category", ""),
                    )

                    if class_name in labels:
                        labels[class_name] = 1

            important_labels = [
                label
                for label in LABEL_COLUMNS
                if label != "no_important_object" and labels[label] == 1
            ]

            if not important_labels:
                labels["no_important_object"] = 1
            else:
                labels["no_important_object"] = 0

            # Reduce near-duplicates and do not flood empty frames.
            should_save = frame_id % save_every == 0

            if labels["no_important_object"] == 1 and random.random() > 0.35:
                should_save = False

            if should_save:
                filename = f"vizdoom_ml_{frame_id:06d}.jpg"
                image_path = IMAGE_DIR / filename

                frame_rgb = state.screen_buffer
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), frame_bgr)

                row = {"filename": filename}
                row.update(labels)
                rows.append(row)

                for label, value in labels.items():
                    counts[label] += int(value)

        game.make_action(choose_action(), 4)

    game.close()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved images to: {IMAGE_DIR}")
    print(f"Saved labels to: {CSV_PATH}")
    print(f"Total saved rows: {len(rows)}")

    print("\nPositive label counts:")
    for label, count in counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()