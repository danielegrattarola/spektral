import numpy as np


class NormalizeSphere:
    def __call__(self, graph):
        offset = np.mean(graph.x, -2, keepdims=True)
        scale = 1 / np.abs(graph.x).max()
        graph.x = (graph.x - offset) * scale

        return graph
