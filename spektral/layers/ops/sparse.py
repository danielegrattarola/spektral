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
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    out = tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
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
        values = a.tocoo().data
        row = a.row
        col = a.col
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


def sparse_add_self_loops(indices, N=None):
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
