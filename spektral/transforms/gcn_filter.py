from spektral.utils import gcn_filter


class GCNFilter(object):
    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = gcn_filter(graph.a, self.symmetric)

        return graph
