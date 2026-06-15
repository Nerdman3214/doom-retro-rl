class YoloDoomDetector:
    def __init__(self, model_path=None, conf_threshold=0.35):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold

        if model_path is None:
            model_path = "yolo11n.pt"

        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(frame, verbose=False, conf=self.conf_threshold)

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            label = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = xyxy
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)

            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "x_center": float((x1 + x2) / 2.0),
                    "y_center": float((y1 + y2) / 2.0),
                    "width": float(width),
                    "height": float(height),
                }
            )

        return detections