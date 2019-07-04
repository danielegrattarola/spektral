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

    ** Usage **

    ```py
    In [1]: from spektral.utils.data import Batch
    Using TensorFlow backend.

    In [2]: A_list = [np.ones((2, 2))] * 3

    In [3]: X_list = [np.random.normal(size=(2, 4))] * 3

    In [4]: b = Batch(A_list, X_list)

    In [5]: b.A.todense()
    Out[5]:
    matrix([[1., 1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]])

    In [6]: b.X
    Out[6]:
    array([[-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
           [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
           [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
           [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
           [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
           [ 0.14221143, -0.76473164, -1.05635638,  1.45961459]])

    In [7]: b.I
    Out[7]: array([0, 0, 1, 1, 2, 2])

    In [8]: b.get('AXI')
    Out[8]:
    (<6x6 sparse matrix of type '<class 'numpy.float64'>'
      with 12 stored elements in COOrdinate format>,
     array([[-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
            [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
            [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
            [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
            [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
            [ 0.14221143, -0.76473164, -1.05635638,  1.45961459]]),
     array([0, 0, 1, 1, 2, 2]))
    ```
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
