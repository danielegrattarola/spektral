from spektral.utils import gcn_filter


class GCNFilter(object):
    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.adj is not None:
            graph.adj = gcn_filter(graph.adj, self.symmetric)

        return graph
