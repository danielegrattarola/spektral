import numpy as np


class NormalizeSphere:
    r"""
        Normalizes the node attributes so that they are centered at the origin and
        contained within a sphere of radius 1:
    $$
            \X_{i} \leftarrow \frac{\X_{i} - \bar\X}{\max_{i,j} \X_{ij}}
        $$

        where \( \bar\X \) is the centroid of the node features.
    """

    def __call__(self, graph):
        offset = np.mean(graph.x, -2, keepdims=True)
        scale = np.abs(graph.x).max()
        graph.x = (graph.x - offset) / scale

        return graph
