"""
Prediction adapter.

This converts backend-specific perception into the shared object/scene format.

ViZDoom already provides real object metadata.
Doom Retro will provide predicted labels from:
    ObjectPredictor
    ScenePredictor
    HUDPredictor
"""

from typing import Any, Dict, List, Optional


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def object_prediction_to_objects(
    object_prediction: Optional[Dict[str, Any]],
    min_confidence: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Convert Doom Retro ObjectPredictor output into shared object dictionaries.

    Expected possible object_prediction shapes:

        {
            "present": ["enemy_visible", "barrel"],
            "scores": {"enemy_visible": 0.82, "barrel": 0.41}
        }

    or:

        {
            "enemy_visible": 0.82,
            "barrel": 0.41
        }

    Output:
        [
            {
                "name": "enemy_visible",
                "category": "enemy",
                "confidence": 0.82,
                "source": "doomretro_object_predictor",
                "position_x": None,
                "position_y": None,
            }
        ]
    """

    if not object_prediction:
        return []

    scores = {}

    if isinstance(object_prediction.get("scores"), dict):
        scores.update(object_prediction.get("scores") or {})

    present = object_prediction.get("present")

    if isinstance(present, list):
        for label in present:
            scores.setdefault(label, 1.0)

    # Fallback: direct label-score dict.
    for key, value in object_prediction.items():
        if key in ["present", "scores", "debug", "raw"]:
            continue

        if isinstance(value, (int, float)):
            scores.setdefault(key, value)

    objects = []

    for label, score in scores.items():
        confidence = _safe_float(score)

        if confidence < min_confidence:
            continue

        label_text = str(label).lower()
        category = predicted_label_to_category(label_text)

        objects.append(
            {
                "name": label_text,
                "category": category,
                "confidence": confidence,
                "source": "doomretro_object_predictor",
                "position_x": None,
                "position_y": None,
            }
        )

    return objects


def scene_prediction_to_scene(
    scene_prediction: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert Doom Retro ScenePredictor output into shared scene_info format.

    Expected possible shapes:

        {
            "label": "front_wall",
            "confidence": 0.88
        }

    or:

        {
            "scene_label": "open_space",
            "scene_confidence": 0.72
        }
    """

    if not scene_prediction:
        return {
            "scene_label": "unknown",
            "scene_confidence": 0.0,
            "source": "none",
        }

    label = (
        scene_prediction.get("scene_label")
        or scene_prediction.get("label")
        or scene_prediction.get("class")
        or "unknown"
    )

    confidence = (
        scene_prediction.get("scene_confidence")
        or scene_prediction.get("confidence")
        or scene_prediction.get("score")
        or 0.0
    )

    return {
        "scene_label": normalize_scene_label(label),
        "scene_confidence": _safe_float(confidence),
        "source": "doomretro_scene_predictor",
        "raw_label": str(label),
    }


def predicted_label_to_category(label: str) -> str:
    """
    Convert predicted Doom Retro labels into shared object categories.
    """

    label = str(label or "").lower()

    if any(token in label for token in ["enemy", "monster", "imp", "zombie", "demon"]):
        return "enemy"

    if any(token in label for token in ["ammo", "clip", "shell", "rocket", "cell"]):
        return "ammo"

    if any(token in label for token in ["health", "stim", "med"]):
        return "health"

    if "armor" in label or "armour" in label:
        return "armor"

    if any(token in label for token in ["weapon", "shotgun", "chaingun", "plasma", "bfg"]):
        return "weapon"

    if "barrel" in label:
        return "barrel"

    if "key" in label or "card" in label or "skull" in label:
        return "key"

    if "door" in label:
        return "door"

    if "switch" in label or "button" in label:
        return "switch"

    return "unknown"


def normalize_scene_label(label: str) -> str:
    """
    Normalize scene predictor labels into SharedDoomLogic labels.
    """

    label = str(label or "").lower()

    if label in ["enemy", "enemy_visible", "combat"]:
        return "enemy_visible"

    if label in ["front_wall", "wall", "blocked", "obstacle", "boundary_or_stuck_wall"]:
        return "front_blocked"

    if label in ["door", "switch", "door_or_switch"]:
        return "door_or_switch"

    if label in ["item", "item_visible", "pickup"]:
        return "item_visible"

    if label in ["open", "open_space", "hallway", "corridor"]:
        return "open_space"

    if label in ["near_wall", "side_wall"]:
        return "near_wall"

    return label


def build_doomretro_game_state(
    base_game_state: Dict[str, Any],
    object_prediction: Optional[Dict[str, Any]] = None,
    scene_prediction: Optional[Dict[str, Any]] = None,
    hud_prediction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a normalized game_state for Doom Retro.

    This does not control Doom Retro.
    It only prepares perception/state data for SharedDoomLogic.
    """

    game_state = dict(base_game_state or {})

    objects = object_prediction_to_objects(object_prediction)
    scene_info = scene_prediction_to_scene(scene_prediction)

    game_state["objects"] = objects
    game_state["external_scene_info"] = scene_info

    if hud_prediction:
        # HUD predictions can fill missing game_state values.
        if game_state.get("health") is None:
            game_state["health"] = hud_prediction.get("health")

        if game_state.get("armor") is None:
            game_state["armor"] = hud_prediction.get("armor")

        if game_state.get("ammo") is None:
            game_state["ammo"] = hud_prediction.get("ammo")

        game_state["hud_prediction"] = hud_prediction

    return game_state
