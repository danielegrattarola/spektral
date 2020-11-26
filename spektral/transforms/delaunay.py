from scipy.spatial import Delaunay as DelaunaySP
import numpy as np
import scipy.sparse as sp


class Delaunay:
    def __call__(self, graph):
        if 'x' not in graph:
            raise ValueError('The graph must have node features')
        if graph.n_node_features != 2:
            raise ValueError('Can only compute triangulation for 2-d points.')
        tri = DelaunaySP(graph.x)
        edges = np.concatenate((tri.vertices[:, :2],
                                tri.vertices[:, 1:],
                                tri.vertices[:, ::2]), axis=0)
        values = np.ones(edges.shape[0])
        graph.a = sp.csr_matrix((values, edges.T))
        graph.a.data = np.clip(graph.a.data, 0, 1)

        return graph
