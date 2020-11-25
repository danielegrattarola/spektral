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
    N = tf.shape(a)[0]
    indices = a.indices
    values = a.values

    mask_od = indices[:, 0] != indices[:, 1]
    mask_sl = ~mask_od

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


def add_self_loops_indices(indices, N=None):
    """
    Given the indices of a square SparseTensor, adds the diagonal entries (i, i)
    and returns the reordered indices.
    :param indices: Tensor of rank 2, the indices to a SparseTensor.
    :param N: the size of the N x N SparseTensor indexed by the indices. If `None`,
    N is calculated as the maximum entry in the indices plus 1.
    :return: Tensor of rank 2, the indices to a SparseTensor.
    """
    N = tf.reduce_max(indices) + 1 if N is None else N
    row, col = indices[..., 0], indices[..., 1]
    mask = tf.ensure_shape(row != col, row.shape)
    sl_indices = tf.range(N, dtype=row.dtype)[:, None]
    sl_indices = tf.repeat(sl_indices, 2, -1)
    indices = tf.concat((indices[mask], sl_indices), 0)
    dummy_values = tf.ones_like(indices[:, 0])
    indices, _ = gen_sparse_ops.sparse_reorder(indices, dummy_values, (N, N))
    return indices


def unsorted_segment_softmax(x, indices, N=None):
    """
    Applies softmax along the segments of a Tensor. This operator is similar
    to the tf.math.segment_* operators, which apply a certain reduction to the
    segments. In this case, the output tensor is not reduced and maintains the
    same shape as the input.
    :param x: a Tensor. The softmax is applied along the first dimension.
    :param indices: a Tensor, indices to the segments.
    :param N: the number of unique segments in the indices. If `None`, N is
    calculated as the maximum entry in the indices plus 1.
    :return: a Tensor with the same shape as the input.
    """
    N = tf.reduce_max(indices) + 1 if N is None else N
    e_x = tf.exp(x - tf.gather(tf.math.unsorted_segment_max(x, indices, N), indices))
    e_x /= tf.gather(tf.math.unsorted_segment_sum(e_x, indices, N) + 1e-9, indices)
    return e_x
