import cv2


class RewardDebugOverlay:

    def draw(self, frame, breakdown):
        y = 25
        for key, value in breakdown.items():
            text = f"{key}: {value:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 20
        return frame
