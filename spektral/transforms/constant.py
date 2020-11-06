import numpy as np


class Constant(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, graph):
        value = np.zeros((graph.N, 1)) + self.value
        if graph.x is None:
            graph.x = value
        else:
            graph.x = np.concatenate((graph.x, value), axis=-1)

        return graph
