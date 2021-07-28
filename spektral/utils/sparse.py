import numpy as np
import tensorflow as tf
from scipy import sparse as sp


def reorder(edge_index, edge_weight=None, edge_features=None):
    """
    Reorders `edge_index`, `edge_weight`, and `edge_features` according to the row-major
    ordering of `edge_index`.
    :param edge_index: np.array of shape `[n_edges, 2]` (representing rows and columns
    of the adjacency matrix), indices to sort in row-major order.
    :param edge_weight: np.array or None, edge weight to sort according to the new
    order of the indices.
    :param edge_features: np.array or None, edge features to sort according to the new
    order of the indices.
    :return:
        - edge_index: np.array, edge_index sorted in row-major order.
        - edge_weight: If edge_weight is not None, edge_weight sorted according to the
        new order of the indices. Otherwise, None.
        - edge_features: If edge_features is not None, edge_features sorted according to
        the new order of the indices. Otherwise, None.
    """
    sort_idx = np.lexsort(np.flipud(edge_index.T))
    output = [edge_index[sort_idx]]
    if edge_weight is not None:
        output.append(edge_weight[sort_idx])
    if edge_features is not None:
        output.append(edge_features[sort_idx])

    return tuple(output)


def edge_index_to_matrix(edge_index, edge_weight, edge_features=None, shape=None):
    reordered = reorder(edge_index, edge_weight, edge_features)
    a = sp.csr_matrix((reordered[1], reordered[0].T), shape=shape)

    if edge_features is not None:
        return a, reordered[2]
    else:
        return a


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    The indices of the output are reordered in the canonical row-major ordering, and
    duplicate entries are summed together (which is the default behaviour of Scipy).

    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    if len(x.shape) != 2:
        raise ValueError("x must have rank 2")
    row, col, values = sp.find(x)
    out = tf.SparseTensor(
        indices=np.array([row, col]).T, values=values, dense_shape=x.shape
    )
    return tf.sparse.reorder(out)


def sp_batch_to_sp_tensor(a_list):
    """
    Converts a list of Scipy sparse matrices to a rank 3 SparseTensor.
    :param a_list: list of Scipy sparse matrices with the same shape.
    :return: SparseTensor of rank 3.
    """
    tensor_data = []
    for i, a in enumerate(a_list):
        row, col, values = sp.find(a)
        batch = np.ones_like(col) * i
        tensor_data.append((values, batch, row, col))
    tensor_data = list(map(np.concatenate, zip(*tensor_data)))

    out = tf.SparseTensor(
        indices=np.array(tensor_data[1:]).T,
        values=tensor_data[0],
        dense_shape=(len(a_list),) + a_list[0].shape,
    )

    return out
