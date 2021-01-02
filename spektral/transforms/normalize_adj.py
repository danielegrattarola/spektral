from spektral.utils import normalized_adjacency


class NormalizeAdj(object):
    r"""
        Normalizes the adjacency matrix as:
    $$
        \A \leftarrow \D^{-1/2}\A\D^{-1/2}
        $$

        **Arguments**

        - `symmetric`: If False, then it computes \(\D^{-1}\A\) instead.
    """

    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = normalized_adjacency(graph.a, self.symmetric)

        return graph
