import numpy as np


class VisionDetector:
    """
    Unified vision interface for DoomEnv.

    Starts with lightweight CV2-style features.
    Later you can plug in:
    - YOLO object detection
    - MobileNetV3 scene classification
    - Depth Anything V2
    - SAM/SAM2-assisted labels
    """

    def __init__(self, frame_processor=None):
        self.frame_processor = frame_processor

        self.use_yolo = False
        self.use_scene_classifier = False
        self.use_depth = False

        self.yolo_model = None
        self.scene_model = None
        self.depth_model = None

    def detect(self, frame):
        result = self._empty_result()

        if frame is None or frame.size == 0:
            return result

        # CV2 fallback signals.
        result.update(self._cv2_fallback(frame))

        # Future YOLO object detector.
        if self.use_yolo and self.yolo_model is not None:
            yolo_result = self._detect_yolo(frame)
            result.update(yolo_result)

        # Future MobileNetV3 scene classifier.
        if self.use_scene_classifier and self.scene_model is not None:
            scene_result = self._classify_scene(frame)
            result.update(scene_result)

        # Future depth model.
        if self.use_depth and self.depth_model is not None:
            depth_result = self._estimate_depth(frame)
            result.update(depth_result)

        return result

    def _empty_result(self):
        return {
            "enemy_visible": False,
            "enemy_centered": False,
            "enemy_left": False,
            "enemy_right": False,
            "enemy_confidence": 0.0,
            "enemy_x": None,
            "enemy_y": None,

            "door_visible": False,
            "door_centered": False,
            "door_left": False,
            "door_right": False,
            "door_confidence": 0.0,

            "pickup_visible": False,
            "weapon_visible": False,
            "ammo_visible": False,
            "health_visible": False,

            "projectile_visible": False,
            "barrel_visible": False,
            "exit_visible": False,

            "front_wall_close": False,
            "left_wall_close": False,
            "right_wall_close": False,
            "front_wall_ratio": 0.0,
            "left_wall_ratio": 0.0,
            "right_wall_ratio": 0.0,

            "scene_label": None,
            "scene_confidence": 0.0,

            "depth_center": None,
            "depth_left": None,
            "depth_right": None,
        }

    def _cv2_fallback(self, frame):
        result = {}

        wall_info = self._wall_direction_info(frame)
        result["left_wall_ratio"] = wall_info["left_ratio"]
        result["front_wall_ratio"] = wall_info["front_ratio"]
        result["right_wall_ratio"] = wall_info["right_ratio"]
        result["left_wall_close"] = wall_info["left_wall"]
        result["front_wall_close"] = wall_info["front_wall"]
        result["right_wall_close"] = wall_info["right_wall"]

        door_error, door_confidence = self._door_horizontal_error(frame)
        result["door_confidence"] = float(door_confidence)

        if door_confidence >= 250:
            result["door_visible"] = True
            result["door_centered"] = abs(door_error) < 0.15
            result["door_left"] = door_error < -0.20
            result["door_right"] = door_error > 0.20

        enemy_error, enemy_confidence = self._enemy_horizontal_error(frame)
        result["enemy_confidence"] = float(enemy_confidence)

        if enemy_confidence >= 20:
            result["enemy_visible"] = True
            result["enemy_centered"] = abs(enemy_error) < 0.18
            result["enemy_left"] = enemy_error < -0.18
            result["enemy_right"] = enemy_error > 0.18
            result["enemy_x"] = float(enemy_error)

        return result

    def _enemy_horizontal_error(self, frame):
        if self.frame_processor is None:
            return 0.0, 0

        if not hasattr(self.frame_processor, "enemy_heatmap"):
            return 0.0, 0

        heatmap = self.frame_processor.enemy_heatmap(frame)

        if heatmap is None or heatmap.size == 0:
            return 0.0, 0

        h, w = heatmap.shape
        gameplay_heatmap = heatmap[: int(h * 0.78), :]

        _, xs = np.where(gameplay_heatmap > 0.5)
        confidence = len(xs)

        if confidence < 8:
            return 0.0, confidence

        enemy_x = float(np.mean(xs))
        center_x = w / 2.0
        error = float((enemy_x - center_x) / center_x)

        return error, confidence

    def _door_horizontal_error(self, frame):
        if frame is None or frame.size == 0:
            return 0.0, 0

        if len(frame.shape) == 3 and frame.shape[0] in [3, 9] and frame.shape[-1] not in [3, 4]:
            frame = np.transpose(frame[:3], (1, 2, 0))

        h, w = frame.shape[:2]
        gameplay = frame[: int(h * 0.70), :]

        blue = gameplay[:, :, 0].astype(np.int16)
        green = gameplay[:, :, 1].astype(np.int16)
        red = gameplay[:, :, 2].astype(np.int16)

        brown = (
            (red > 55)
            & (green > 35)
            & (blue < 120)
            & (red >= green)
        )

        gray = (
            (red > 50)
            & (green > 50)
            & (blue > 50)
            & (np.abs(red - green) < 25)
            & (np.abs(green - blue) < 25)
        )

        mask = brown | gray

        center_left = int(w * 0.20)
        center_right = int(w * 0.80)

        mask[:, :center_left] = False
        mask[:, center_right:] = False

        _, xs = np.where(mask)
        confidence = len(xs)

        if confidence < 20:
            return 0.0, confidence

        door_x = float(np.mean(xs))
        center_x = w / 2.0
        error = float((door_x - center_x) / center_x)

        return error, confidence

    def _wall_direction_info(self, frame):
        default = {
            "left_ratio": 0.0,
            "front_ratio": 0.0,
            "right_ratio": 0.0,
            "left_wall": False,
            "front_wall": False,
            "right_wall": False,
        }

        if frame is None or frame.size == 0:
            return default

        if len(frame.shape) == 3 and frame.shape[0] in [3, 9] and frame.shape[-1] not in [3, 4]:
            frame = np.transpose(frame[:3], (1, 2, 0))

        h, w = frame.shape[:2]
        gameplay = frame[: int(h * 0.70), :]

        blue = gameplay[:, :, 0].astype(np.int16)
        green = gameplay[:, :, 1].astype(np.int16)
        red = gameplay[:, :, 2].astype(np.int16)

        brown = (
            (red > 45)
            & (green > 25)
            & (blue < 130)
            & (red >= green)
            & (green >= blue * 0.45)
        )

        gray = (
            (red > 45)
            & (green > 45)
            & (blue > 45)
            & (np.abs(red - green) < 30)
            & (np.abs(green - blue) < 30)
        )

        wall_mask = brown | gray

        left_region = wall_mask[:, : int(w * 0.33)]
        front_region = wall_mask[:, int(w * 0.33): int(w * 0.66)]
        right_region = wall_mask[:, int(w * 0.66):]

        left_ratio = float(np.mean(left_region))
        front_ratio = float(np.mean(front_region))
        right_ratio = float(np.mean(right_region))

        return {
            "left_ratio": left_ratio,
            "front_ratio": front_ratio,
            "right_ratio": right_ratio,
            "left_wall": left_ratio > 0.42,
            "front_wall": front_ratio > 0.42,
            "right_wall": right_ratio > 0.42,
        }

    def _detect_yolo(self, frame):
        # Placeholder for Phase 2.
        return {}

    def _classify_scene(self, frame):
        # Placeholder for Phase 3.
        return {}

    def _estimate_depth(self, frame):
        # Placeholder for Phase 4.
        return {}