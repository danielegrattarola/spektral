import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from tensorflow.keras.datasets import mnist as m

from spektral.data import Dataset, Graph

MNIST_SIZE = 28


class MNIST(Dataset):
    """
    The MNIST images used as node features for a grid graph, as described by
    [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).

    This dataset is a graph signal classification task, where graphs are
    represented in mixed mode: one adjacency matrix, many instances of node
    features.

    For efficiency, the adjacency matrix is stored in a special attribute of the
    dataset and the Graphs only contain the node features.
    You can access the adjacency matrix via the `a` attribute.

    The node features of each graph are the MNIST digits vectorized and rescaled
    to [0, 1].
    Two nodes are connected if they are neighbours on the grid.
    Labels represent the MNIST class associated to each sample.

    **Note:** the last 10000 samples are the default test set of the MNIST
    dataset.

    **Arguments**

    - `p_flip`: if >0, then edges are randomly flipped from 0 to 1 or vice versa
    with that probability.
    - `k`: number of neighbours of each node.
    """

    def __init__(self, p_flip=0.0, k=8, **kwargs):
        self.a = None
        self.k = k
        self.p_flip = p_flip
        super().__init__(**kwargs)

    def read(self):
        self.a = _mnist_grid_graph(self.k)
        self.a = _flip_random_edges(self.a, self.p_flip)

        (x_train, y_train), (x_test, y_test) = m.load_data()
        x = np.vstack((x_train, x_test))
        x = x / 255.0
        y = np.concatenate((y_train, y_test), 0)
        x = x.reshape(-1, MNIST_SIZE ** 2, 1)

        return [Graph(x=x_, y=y_) for x_, y_ in zip(x, y)]


def _grid_coordinates(side):
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def _get_adj_from_data(X, k, **kwargs):
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A


def _mnist_grid_graph(k):
    X = _grid_coordinates(MNIST_SIZE)
    A = _get_adj_from_data(
        X, k, mode="connectivity", metric="euclidean", include_self=False
    )

    return A


def _flip_random_edges(A, p_swap):
    if not A.shape[0] == A.shape[1]:
        raise ValueError("A must be a square matrix.")
    dtype = A.dtype
    A = sp.lil_matrix(A).astype(np.bool)
    n_elem = A.shape[0] ** 2
    n_elem_to_flip = round(p_swap * n_elem)
    unique_idx = np.random.choice(n_elem, replace=False, size=n_elem_to_flip)
    row_idx = unique_idx // A.shape[0]
    col_idx = unique_idx % A.shape[0]
    idxs = np.stack((row_idx, col_idx)).T
    for i in idxs:
        i = tuple(i)
        A[i] = np.logical_not(A[i])
    A = A.tocsr().astype(dtype)
    A.eliminate_zeros()
    return A
