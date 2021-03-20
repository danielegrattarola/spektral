import numpy as np
from scipy.sparse.linalg import eigsh

from spektral.utils import normalized_laplacian


class LaplacianPE:
    r"""
    Adds Laplacian positional encodings to the nodes.

    The first `k` eigenvectors are computed and concatenated to the node features.
    Each node will be extended with its corresponding entries in the first k
    eigenvectors.
    """

    def __init__(self, k):
        assert k > 0, "k must be greater than 0"
        self.k = k

    def __call__(self, graph):
        if "a" not in graph:
            raise ValueError("The graph must have an adjacency matrix")
        assert (
            self.k < graph.n_nodes
        ), f"k = {self.k} must be smaller than graph.n_nodes = {graph.n_nodes}"

        l = normalized_laplacian(graph.a)
        _, eigvec = eigsh(l, k=self.k, which="SM")

        if "x" not in graph:
            graph.x = eigvec
        else:
            graph.x = np.concatenate((graph.x, eigvec), axis=-1)

        return graph
