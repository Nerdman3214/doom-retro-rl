import numpy as np


class ExplorationMemory:

    def __init__(self, map_size=512):
        self.map_size = map_size
        self.heatmap = np.zeros((map_size, map_size), dtype=np.float32)

    def visit(self, x, y):
        x = int(np.clip(x, 0, self.map_size - 1))
        y = int(np.clip(y, 0, self.map_size - 1))
        self.heatmap[y, x] += 1

    def get_novelty(self, x, y):
        x = int(np.clip(x, 0, self.map_size - 1))
        y = int(np.clip(y, 0, self.map_size - 1))
        visits = self.heatmap[y, x]
        return 1.0 / (1.0 + visits)

    def get_heatmap(self):
        return self.heatmap
