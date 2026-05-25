import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import vizdoom as vzd
except Exception as e:
    raise RuntimeError(f"ViZDoom import failed: {e}")


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vision_dataset_v2" / "vizdoom_geometry"
IWAD = "/usr/share/games/doom/freedoom1.wad"


def ensure_hwc_rgb(screen):
    arr = np.asarray(screen)

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    return arr.astype(np.uint8)


def normalize_depth(depth):
    if depth is None:
        return None

    arr = np.asarray(depth).astype(np.float32)

    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)

    values = arr[finite]
    lo = float(values.min())
    hi = float(values.max())

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    norm = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    return norm


def labels_to_uint8(labels):
    if labels is None:
        return None

    arr = np.asarray(labels)

    if arr.ndim == 3:
        arr = arr[0]

    return arr.astype(np.uint8)


def automap_to_rgb(automap):
    if automap is None:
        return None

    arr = np.asarray(automap)

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    return arr.astype(np.uint8)


def make_geometry_mask(depth_uint8, labels_uint8):
    """
    Simple geometry/obstacle mask.

    This is not perfect semantic segmentation. It gives you a teacher label
    for visible near geometry:
    - near visible surfaces from depth
    - labels buffer nonzero regions

    Output:
        0 = background/far/unknown
        1 = visible geometry/object/obstacle-ish
        2 = very near/front-blocking geometry
    """
    if depth_uint8 is None:
        return None

    depth = depth_uint8.astype(np.uint8)

    # Per-frame normalized depth:
    # high values are farther after normalization, low values are closer.
    near = depth < 70
    mid = depth < 120

    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[mid] = 1
    mask[near] = 2

    if labels_uint8 is not None:
        label_nonzero = labels_uint8 > 0
        mask[label_nonzero & (mask == 0)] = 1

    return mask


def object_to_dict(obj):
    output = {}

    for attr in [
        "id",
        "name",
        "position_x",
        "position_y",
        "position_z",
        "angle",
        "pitch",
        "roll",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "width",
        "height",
    ]:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            try:
                if isinstance(value, (int, float, str, bool)):
                    output[attr] = value
                else:
                    output[attr] = float(value)
            except Exception:
                output[attr] = str(value)

    return output


def sector_to_dict(sec):
    output = {}

    for attr in [
        "floor_height",
        "ceiling_height",
        "line_count",
    ]:
        if hasattr(sec, attr):
            try:
                output[attr] = float(getattr(sec, attr))
            except Exception:
                output[attr] = str(getattr(sec, attr))

    return output


def setup_game(visible=False):
    game = vzd.DoomGame()

    if os.path.exists(IWAD):
        game.set_doom_game_path(IWAD)

    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_available_buttons(
        [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.USE,
        ]
    )

    game.set_available_game_variables(
        [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.ITEMCOUNT,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.ANGLE,
        ]
    )

    game.init()
    return game


def random_action():
    # button order:
    # forward, backward, turn_left, turn_right, strafe_left, strafe_right, attack, use
    actions = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ]

    return random.choice(actions)


def collect(num_steps=2000, visible=False, frame_skip=4):
    OUT.mkdir(parents=True, exist_ok=True)

    dirs = {
        "rgb": OUT / "rgb",
        "depth": OUT / "depth",
        "labels": OUT / "labels",
        "automap": OUT / "automap",
        "geometry_mask": OUT / "geometry_mask",
        "meta": OUT / "meta",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    game = setup_game(visible=visible)
    game.new_episode()

    saved = 0

    try:
        for step in range(num_steps):
            if game.is_episode_finished():
                game.new_episode()

            state = game.get_state()

            if state is not None:
                rgb = ensure_hwc_rgb(state.screen_buffer)
                depth = normalize_depth(getattr(state, "depth_buffer", None))
                labels = labels_to_uint8(getattr(state, "labels_buffer", None))
                automap = automap_to_rgb(getattr(state, "automap_buffer", None))
                geom_mask = make_geometry_mask(depth, labels)

                name = f"{saved:07d}"

                cv2.imwrite(str(dirs["rgb"] / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if depth is not None:
                    cv2.imwrite(str(dirs["depth"] / f"{name}.png"), depth)

                if labels is not None:
                    cv2.imwrite(str(dirs["labels"] / f"{name}.png"), labels)

                if automap is not None:
                    cv2.imwrite(str(dirs["automap"] / f"{name}.png"), cv2.cvtColor(automap, cv2.COLOR_RGB2BGR))

                if geom_mask is not None:
                    cv2.imwrite(str(dirs["geometry_mask"] / f"{name}.png"), geom_mask)

                objects = []
                if getattr(state, "objects", None) is not None:
                    objects = [object_to_dict(obj) for obj in state.objects]

                sectors = []
                if getattr(state, "sectors", None) is not None:
                    sectors = [sector_to_dict(sec) for sec in state.sectors]

                vars_list = []
                if state.game_variables is not None:
                    vars_list = [float(v) for v in state.game_variables]

                meta = {
                    "step": step,
                    "file_id": name,
                    "game_variables": vars_list,
                    "objects": objects,
                    "sectors": sectors,
                }

                with open(dirs["meta"] / f"{name}.json", "w") as f:
                    json.dump(meta, f, indent=2)

                saved += 1

                if saved % 100 == 0:
                    print(f"[vizdoom_geometry] saved={saved}")

            game.make_action(random_action(), frame_skip)

    finally:
        game.close()

    print(f"[vizdoom_geometry] done saved={saved} out={OUT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    collect(
        num_steps=args.steps,
        visible=args.visible,
        frame_skip=args.frame_skip,
    )
