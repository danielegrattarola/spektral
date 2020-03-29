import scipy.sparse as sp
import numpy as np


def graph_to_numpy(graph, dtype=None):
    """
    Converts a graph in OGB's library-agnostic format to a representation in
    Numpy/Scipy. See the [Open Graph Benchmark's website](https://ogb.stanford.edu)
    for more information.
    :param graph: OGB library-agnostic graph;
    :param dtype: if set, all output arrays will be cast to this dtype.
    :return:
        - X: np.array of shape (N, F) with the node features;
        - A: scipy.sparse adjacency matrix of shape (N, N) in COOrdinate format;
        - E: if edge features are available, np.array of shape (n_edges, S),
            `None` otherwise.
    """
    N = graph['num_nodes']
    X = graph['node_feat'].astype(dtype)
    row, col = graph['edge_index']
    A = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(N, N)).astype(dtype)
    E = graph['edge_feat'].astype(dtype)

    return X, A, E


def dataset_to_numpy(dataset, indices=None, dtype=None):
    """
    Converts a dataset in OGB's library-agnostic version to lists of Numpy/Scipy
    arrays. See the [Open Graph Benchmark's website](https://ogb.stanford.edu)
    for more information.
    :param dataset: OGB library-agnostic dataset (e.g., GraphPropPredDataset);
    :param indices: optional, a list of integer indices; if provided, only these
    graphs will be converted;
    :param dtype: if set, the arrays in the returned lists will have this dtype.
    :return:
        - X_list: list of np.arrays of (variable) shape (N, F) with node features;
        - A_list: list of scipy.sparse adjacency matrices of (variable) shape
        (N, N);
        - E_list: list of np.arrays of (variable) shape (n_nodes, S) with edge
        attributes. If edge attributes are not available, a list of None.
        - y_list: np.array of shape (n_graphs, n_tasks) with the task labels;
    """
    X_list = []
    A_list = []
    E_list = []
    y_list = []
    if indices is None:
        indices = range(len(dataset))

    for i in indices:
        graph, label = dataset[int(i)]
        X, A, E = graph_to_numpy(graph, dtype=dtype)
        X_list.append(X)
        A_list.append(A)
        E_list.append(E)
        y_list.append(label)

    return X_list, A_list, E_list, np.array(y_list)