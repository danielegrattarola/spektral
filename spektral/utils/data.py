import numpy as np
import scipy.sparse as sp


class Batch(object):
    """
    Converts a batch of graphs stored in lists to the graph batch format
    (see [https://danielegrattarola.github.io/spektral/data/](https://danielegrattarola.github.io/spektral/data/)).

    **Input**

    - A_list, list of adjacency matrices of shape `(N, N)`;
    - X_list, list of node attributes matrices of shape `(N, F)`;

    ** Properties **

    - A: returns the block diagonal adjacency matrix;
    - X: returns the stacked node attributes;
    - I: returns the graph indices mapping each node to a graph (numbering is
    relative to the batch).

    """
    def __init__(self, A_list, X_list):
        self.A_list = A_list
        self.X_list = X_list

        n_nodes = np.array([a_.shape[0] for a_ in self.A_list])
        self.I_list = np.repeat(np.arange(len(n_nodes)), n_nodes)

        self.attr_map = {'A': self.A,
                         'X': self.X,
                         'I': self.I}

    def get(self, attr):
        """
        Splits a strings and returns the associated matrices in order, e.g.,
        `data = b.get('AXI')` is equivalent to `data = b.A, b.X, b.I`.
        :param attr: a string (possible literals can be seen with `b.attr_map.keys()`.
        :return: a tuple with the requested matrices.
        """
        try:
            return tuple(self.attr_map[a] for a in list(attr))
        except KeyError:
            raise KeyError('Possible attributes: {}'.format(self.attr_map.keys()))

    @property
    def A(self):
        return sp.block_diag(self.A_list)

    @property
    def X(self):
        return np.vstack(self.X_list)

    @property
    def I(self):
        return self.I_list