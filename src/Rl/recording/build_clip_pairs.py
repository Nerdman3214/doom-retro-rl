import os
import random
import pickle


clip_dir = "clips"
clips = os.listdir(clip_dir)


pairs = []


for _ in range(len(clips) // 2):

    a, b = random.sample(clips, 2)

    pairs.append((a, b))


with open("clip_pairs.pkl", "wb") as f:

    pickle.dump(pairs, f)


print("Clip pairs generated.")