import numpy as np


class NormalizeOne:
    r"""
    Normalizes the node attributes by dividing each row by its sum, so that it
    sums to 1:
    $$
        \X_i \leftarrow \frac{\X_i}{\sum_{j=1}^{N} \X_{ij}}
    $$

    """

    def __call__(self, graph):
        x_sum = np.sum(graph.x, -1)
        x_sum[x_sum == 0] = 1
        graph.x = graph.x / x_sum[..., None]

        return graph
