import numpy as np
import scipy.sparse as sp
from keras.datasets import mnist as m
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph


def load_data():
    """
    Loads the MNIST dataset and the associated grid.
    This code is largely taken from [Michaël Defferrard's Github](https://github.com/mdeff/cnn_graph/blob/master/nips2016/mnist.ipynb).

    :return:
        - X_train, y_train: training node features and labels;
        - X_val, y_val: validation node features and labels;
        - X_test, y_test: test node features and labels;
        - A: adjacency matrix of the grid;
    """

    A = grid_graph(28, corners=False)
    A = replace_random_edges(A, 0).astype(np.float32)

    (X_train, y_train), (X_test, y_test) = m.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000)

    return X_train, y_train, X_val, y_val, X_test, y_test, A


def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = sp.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def eval_bw(X, Y):
    """
    Compute heuristically the bandwidth using class information
    Returns (d^2)/9, with d minimum distance of elements in X with different class Y
    A small value is added to avoid returning bw=0
    """
    classes = np.unique(Y)
    min_dist = np.inf
    for i in range(classes.shape[0] - 1):
        c_i = classes[i]
        X_i = X[Y == c_i, :]
        for j in range(i + 1, classes.shape[0]):
            c_j = classes[j]
            X_j = X[Y == c_j, :]
            dist_ij = np.min(cdist(X_i, X_j, metric='sqeuclidean'))
            if dist_ij < min_dist:
                min_dist = dist_ij

    return min_dist / 9.0 + 1e-6


def _grid(m, dtype=np.float32):
    """Returns the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def get_adj_from_data(X_l, Y_l=None, X_u=None, adj='knn', k=10, knn_mode='distance', metric='euclidean',
                      self_conn=True):
    """
    :param X_l: labelled node features;
    :param Y_l: labels associated to X_l;
    :param X_u: unlabelled node features;
    :param adj: type of adjacency matrix to compute.
        - 'rbf' to compute rbf with bandwidth evaluated heuristically with
        eval_bw;
        - 'knn' to compute a kNN graph (k must be specified)
    :param k: number of neighbors in the kNN graph or in the linear neighborhood;
    :param knn_mode: 'connectivity' (graph with 0 and 1) or 'distance';
    :param metric: metric to use to build the knn graph (see sklearn.neighbors.kneighbors_graph)
    :param self_conn: if True, self connections are removed from adj matrix (A_ii = 0)
    :return: adjacency matrix as a sparse array (knn) or numpy array (rbf)
    """
    if adj not in {'rbf', 'knn'}:
        raise ValueError('adj must be either rbf or knn')
    if X_u is not None:
        X = np.concatenate((X_l, X_u), axis=0)
    else:
        X = X_l

    # Compute transition prob matrix
    if adj == 'rbf':
        # Estimate bandwidth
        if Y_l is None:
            bw = 0.01
        else:
            bw = eval_bw(X_l, np.argmax(Y_l, axis=1))

        # Compute adjacency matrix
        d = squareform(pdist(X, metric='sqeuclidean'))
        A = np.exp(-d / bw)

        # No self-connections (avoids self-reinforcement)
        if self_conn is False:
            np.fill_diagonal(A, 0.0)
    elif adj == 'knn':
        if k is None:
            raise ValueError('k must be specified when adj=\'knn\'')
        # Compute adjacency matrix
        A = kneighbors_graph(
            X, n_neighbors=k,
            mode=knn_mode,
            metric=metric,
            include_self=self_conn
        ).toarray()
        A = sp.csr_matrix(np.maximum(A, A.T))
    else:
        raise NotImplementedError()

    return A


def grid_graph(m, corners=False):
    z = _grid(m)
    A = get_adj_from_data(z, adj='knn', k=8, metric='euclidean')

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = sp.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    return A
