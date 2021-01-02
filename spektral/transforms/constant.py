import numpy as np


class Constant(object):
    """
    Concatenates a constant value to the node attributes.

    **Arguments**

    - `value`: the value to concatenate to the node attributes.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, graph):
        value = np.zeros((graph.n_nodes, 1)) + self.value
        if graph.x is None:
            graph.x = value
        else:
            graph.x = np.concatenate((graph.x, value), axis=-1)

        return graph
