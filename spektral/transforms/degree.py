import numpy as np

from spektral.utils import one_hot


class Degree(object):
    """
    Concatenates to each node attribute the one-hot degree of the corresponding
    node.

    The adjacency matrix is expected to have integer entries and the degree is
    cast to integer before one-hot encoding.

    **Arguments**

    - `max_degree`: the maximum degree of the nodes, i.e., the size of the
    one-hot vectors.
    """

    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, graph):
        if "a" not in graph:
            raise ValueError("The graph must have an adjacency matrix")
        degree = graph.a.sum(1).astype(int)
        if isinstance(degree, np.matrix):
            degree = np.asarray(degree)[:, 0]
        degree = one_hot(degree, self.max_degree + 1)
        if "x" not in graph:
            graph.x = degree
        else:
            graph.x = np.concatenate((graph.x, degree), axis=-1)

        return graph


class MaxDegree(object):
    def __call__(self, graph):
        return graph.a.sum(1).max()
