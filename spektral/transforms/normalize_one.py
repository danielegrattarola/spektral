import numpy as np


class NormalizeOne:
    def __call__(self, graph):
        x_sum = np.sum(graph.x, -1)
        x_sum[x_sum == 0] = 1
        graph.x = graph.x / x_sum[..., None]

        return graph
