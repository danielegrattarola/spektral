import networkx as nx
import numpy as np


class ClusteringCoeff:
    """
    Concatenates to each node attribute the clustering coefficient of the
    corresponding node.
    """

    def __call__(self, graph):
        if "a" not in graph:
            raise ValueError("The graph must have an adjacency matrix")
        clustering_coeff = nx.clustering(nx.Graph(graph.a))
        clustering_coeff = np.array(
            [clustering_coeff[i] for i in range(graph.n_nodes)]
        )[:, None]

        if "x" not in graph:
            graph.x = clustering_coeff
        else:
            graph.x = np.concatenate((graph.x, clustering_coeff), axis=-1)

        return graph
