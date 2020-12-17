import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from tensorflow.python.ops import gen_sparse_ops


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    if len(x.shape) != 2:
        raise ValueError('x must have rank 2')
    row, col, values = sp.find(x)
    out = tf.SparseTensor(
        indices=np.array([row, col]).T,
        values=values,
        dense_shape=x.shape
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
        dense_shape=(len(a_list), ) + a_list[0].shape
    )

    return out


def dense_to_sparse(x):
    """
    Converts a Tensor to a SparseTensor.
    :param x: a Tensor.
    :return: a SparseTensor.
    """
    indices = tf.where(tf.not_equal(x, 0))
    values = tf.gather_nd(x, indices)
    shape = tf.shape(x, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def add_self_loops(a, fill=1.):
    """
    Adds self-loops to the given adjacency matrix. Self-loops are added only for
    those node that don't have a self-loop already, and are assigned a weight
    of `fill`.
    :param a: a square SparseTensor.
    :param fill: the fill value for the new self-loops. It will be cast to the
    dtype of `a`.
    :return: a SparseTensor with the same shape as the input.
    """
    indices = a.indices
    values = a.values
    N = tf.shape(a, out_type=indices.dtype)[0]

    mask_od = indices[:, 0] != indices[:, 1]
    mask_sl = ~mask_od
    mask_od.set_shape([None])  # For compatibility with TF 2.2
    mask_sl.set_shape([None])

    indices_od = indices[mask_od]
    indices_sl = indices[mask_sl]

    values_sl = tf.fill((N, ), tf.cast(fill, values.dtype))
    values_sl = tf.tensor_scatter_nd_update(
        values_sl, indices_sl[:, 0:1], values[mask_sl])

    indices_sl = tf.range(N, dtype=indices.dtype)[:, None]
    indices_sl = tf.repeat(indices_sl, 2, -1)
    indices = tf.concat((indices_od, indices_sl), 0)

    values_od = values[mask_od]
    values = tf.concat((values_od, values_sl), 0)

    out = tf.SparseTensor(indices, values, (N, N))

    return tf.sparse.reorder(out)


def add_self_loops_indices(indices, n_nodes=None):
    """
    Given the indices of a square SparseTensor, adds the diagonal entries (i, i)
    and returns the reordered indices.
    :param indices: Tensor of rank 2, the indices to a SparseTensor.
    :param n_nodes: the size of the n_nodes x n_nodes SparseTensor indexed by
    the indices. If `None`, n_nodes is calculated as the maximum entry in the
    indices plus 1.
    :return: Tensor of rank 2, the indices to a SparseTensor.
    """
    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
    row, col = indices[..., 0], indices[..., 1]
    mask = tf.ensure_shape(row != col, row.shape)
    sl_indices = tf.range(n_nodes, dtype=row.dtype)[:, None]
    sl_indices = tf.repeat(sl_indices, 2, -1)
    indices = tf.concat((indices[mask], sl_indices), 0)
    dummy_values = tf.ones_like(indices[:, 0])
    indices, _ = gen_sparse_ops.sparse_reorder(indices, dummy_values, (n_nodes, n_nodes))
    return indices
