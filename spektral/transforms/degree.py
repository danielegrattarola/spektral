import numpy as np

from spektral.utils import one_hot


class Degree(object):
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, graph):
        degree = graph.adj.sum(1)
        degree = one_hot(degree, self.max_degree + 1)
        if graph.x is None:
            graph.x = degree
        else:
            graph.x = np.concatenate((graph.x, degree), axis=-1)

        return graph


class MaxDegree(object):
    def __call__(self, graph):
        return graph.adj.sum(1).max()
