import numpy as np
import scipy.sparse as sp


def reorder(edge_index, edge_weight=None, edge_features=None):
    """
    Sorts index in lexicographic order and reorders data accordingly.
    :param edge_index: np.array, indices to sort in lexicographic order.
    :param edge_weight: np.array or None, edge weight to sort according to the new
    order of the indices.
    :param edge_features: np.array or None, edge features to sort according to the new
    order of the indices.
    :return:
        - edge_index: np.array, edge_index sorted in lexicographic order.
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
