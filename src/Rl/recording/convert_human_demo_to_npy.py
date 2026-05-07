import pickle
import numpy as np
import os
from os.path import dirname, abspath

RL_ROOT = dirname(dirname(abspath(__file__)))


def main():
    input_file = os.path.join(RL_ROOT, "human_demo.pkl")
    output_dir = os.path.join(RL_ROOT, "datasets", "human_demos")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(input_file, "rb") as f:
        unpickler = pickle.Unpickler(f)
        while True:
            try:
                frame, action = unpickler.load()
            except EOFError:
                break
            np.savez_compressed(
                os.path.join(output_dir, f"sample_human_{count}.npz"),
                frame=frame,
                action=str(action),
                health=100,
                ammo=50,
            )
            count += 1

    print(f"Converted {count} frames to npz in {output_dir}")


if __name__ == "__main__":
    main()

