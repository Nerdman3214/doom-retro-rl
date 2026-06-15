import pickle
import cv2


def play_clip(path):
    with open(path, "rb") as f:
        clip = pickle.load(f)

    for frame, action in clip:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        big = cv2.resize(frame_bgr, (336, 336),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, str(action), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Clip Preview", big)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "clips/clip_0.pkl"
    play_clip(path)
