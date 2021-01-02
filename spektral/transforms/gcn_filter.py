from spektral.utils import gcn_filter


class GCNFilter(object):
    r"""
        Normalizes the adjacency matrix as described by
        [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907):
    $$
        \A \leftarrow \hat\D^{-\frac{1}{2}} (\A + \I) \hat\D^{-\frac{1}{2}}
        $$

        where \( \hat\D_{ii} = 1 + \sum\limits_{j = 1}^{N} \A_{ij} \).

        **Arguments**

        - `symmetric`: If False, then it computes \(\hat\D^{-1} (\A + \I)\) instead.
    """

    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = gcn_filter(graph.a, self.symmetric)

        return graph
