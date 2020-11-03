from spektral.utils import normalized_adjacency


class NormalizeAdj(object):
    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.adj is not None:
            graph.adj = normalized_adjacency(graph.adj, self.symmetric)

        return graph
