from spektral.utils import normalized_adjacency


class NormalizeAdj(object):
    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = normalized_adjacency(graph.a, self.symmetric)

        return graph
