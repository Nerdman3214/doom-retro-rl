import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.prediction_adapter import (
    object_prediction_to_objects,
    scene_prediction_to_scene,
    build_doomretro_game_state,
)
from env.shared_doom_logic import SharedDoomLogic


def main():
    object_prediction = {
        "present": ["enemy_visible", "barrel", "health_pickup"],
        "scores": {
            "enemy_visible": 0.82,
            "barrel": 0.45,
            "health_pickup": 0.76,
            "ammo_pickup": 0.22,
        },
    }

    scene_prediction = {
        "label": "front_wall",
        "confidence": 0.88,
    }

    base_game_state = {
        "health": 74,
        "armor": 10,
        "ammo": 4,
        "kill_count": 0,
        "item_count": 2,
        "x": 100,
        "y": 200,
        "z": 0,
        "angle": 90,
    }

    objects = object_prediction_to_objects(object_prediction)
    scene = scene_prediction_to_scene(scene_prediction)

    game_state = build_doomretro_game_state(
        base_game_state=base_game_state,
        object_prediction=object_prediction,
        scene_prediction=scene_prediction,
        hud_prediction=None,
    )

    logic = SharedDoomLogic(level_name="freedoom_e1m1")

    object_summary = logic.summarize_objects(game_state)
    inferred_scene = logic.infer_scene(
        game_state,
        depth_info={"front_blocked": True, "near_wall": True},
        object_summary=object_summary,
    )

    print("[adapter_test] objects:", objects)
    print("[adapter_test] scene:", scene)
    print("[adapter_test] game_state:", game_state)
    print("[adapter_test] object_summary:", object_summary)
    print("[adapter_test] inferred_scene:", inferred_scene)


if __name__ == "__main__":
    main()
