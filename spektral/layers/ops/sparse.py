import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from tensorflow.python.ops import gen_sparse_ops

from . import ops


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
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


def add_self_loops(a, fill=1.0):
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

    values_sl = tf.fill((N,), tf.cast(fill, values.dtype))
    values_sl = tf.tensor_scatter_nd_update(
        values_sl, indices_sl[:, 0:1], values[mask_sl]
    )

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
    indices, _ = gen_sparse_ops.sparse_reorder(
        indices, dummy_values, (n_nodes, n_nodes)
    )
    return indices


def _square_size(dense_shape):
    dense_shape = tf.unstack(dense_shape)
    size = dense_shape[0]
    for d in dense_shape[1:]:
        tf.debugging.assert_equal(size, d)
    return d


def _indices_to_inverse_map(indices, size):
    """
    Compute inverse indices of a gather.
    :param indices: Tensor, forward indices, rank 1
    :param size: Tensor, size of pre-gathered input, rank 0
    :return: Tensor, inverse indices, shape [size]. Zero values everywhere
    except at indices.
    """
    indices = tf.cast(indices, tf.int64)
    size = tf.cast(size, tf.int64)
    return tf.scatter_nd(
        tf.expand_dims(indices, axis=-1),
        tf.range(tf.shape(indices, out_type=tf.int64)[0]),
        tf.expand_dims(size, axis=-1),
    )


def _boolean_mask_sparse(a, mask, axis, inverse_map, out_size):
    """
    SparseTensor equivalent to tf.boolean_mask.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param mask: rank-1 bool Tensor.
    :param axis: int, axis on which to mask. Must be in [-k, k).
    :param out_size: number of true entires in mask. Computed if not given.
    :return masked_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving edges, shape [nnz].
    """
    mask = tf.convert_to_tensor(mask)
    values_mask = tf.gather(mask, a.indices[:, axis], axis=0)
    dense_shape = tf.tensor_scatter_nd_update(a.dense_shape, [[axis]], [out_size])
    indices = tf.boolean_mask(a.indices, values_mask)
    indices = tf.unstack(indices, axis=-1)
    indices[axis] = tf.gather(inverse_map, indices[axis])
    indices = tf.stack(indices, axis=-1)
    a = tf.SparseTensor(
        indices,
        tf.boolean_mask(a.values, values_mask),
        dense_shape,
    )
    return (a, values_mask)


def _boolean_mask_sparse_square(a, mask, inverse_map, out_size):
    """
    Apply boolean_mask to every axis of a SparseTensor.
    :param a: SparseTensor with uniform dimensions and nnz non-zeros.
    :param mask: boolean mask.
    :param inverse_map: Tensor of new indices, shape [nnz]. Computed if None.
    :out_size: number of True values in mask. Computed if None.
    :return a: SparseTensor with uniform dimensions.
    :return values_mask: bool Tensor of shape [nnz] indicating valid edges.
    """
    mask = tf.convert_to_tensor(mask)
    values_mask = tf.reduce_all(tf.gather(mask, a.indices, axis=0), axis=-1)
    dense_shape = [out_size] * a.shape.ndims
    indices = tf.boolean_mask(a.indices, values_mask)
    indices = tf.gather(inverse_map, indices)
    a = tf.SparseTensor(indices, tf.boolean_mask(a.values, values_mask), dense_shape)
    return (a, values_mask)


def boolean_mask_sparse(a, mask, axis=0):
    """
    SparseTensor equivalent to tf.boolean_mask.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param mask: rank-1 bool Tensor.
    :param axis: int, axis on which to mask. Must be in [-k, k).
    :return masked_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    i = tf.squeeze(tf.where(mask), axis=1)
    out_size = tf.math.count_nonzero(mask)
    in_size = a.dense_shape[axis]
    inverse_map = _indices_to_inverse_map(i, in_size)
    return _boolean_mask_sparse(
        a, mask, axis=axis, inverse_map=inverse_map, out_size=out_size
    )


def boolean_mask_sparse_square(a, mask):
    """
    Apply mask to every axis of SparseTensor a.
    :param a: SparseTensor, square, nnz non-zeros.
    :param mask: boolean mask with size equal to each dimension of a.
    :return masked_a: SparseTensor
    :return values_mask: bool tensor of shape [nnz] indicating valid values.
    """
    i = tf.squeeze(tf.where(mask), axis=-1)
    out_size = tf.size(i)
    in_size = _square_size(a.dense_shape)
    inverse_map = _indices_to_inverse_map(i, in_size)
    return _boolean_mask_sparse_square(
        a, mask, inverse_map=inverse_map, out_size=out_size
    )


def gather_sparse(a, indices, axis=0, mask=None):
    """
    SparseTensor equivalent to tf.gather, assuming indices are sorted.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param indices: rank-1 int Tensor, rows or columns to keep.
    :param axis: int axis to apply gather to.
    :param mask: boolean mask corresponding to indices. Computed if not provided.
    :return gathered_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    in_size = _square_size(a.dense_shape)
    out_size = tf.size(indices)
    if mask is None:
        mask = ops.indices_to_mask(indices, in_size)
    inverse_map = _indices_to_inverse_map(indices, in_size)
    return _boolean_mask_sparse(
        a, mask, axis=axis, inverse_map=inverse_map, out_size=out_size
    )


def gather_sparse_square(a, indices, mask=None):
    """
    Gather on every axis of a SparseTensor.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param indices: rank-1 int Tensor, rows and columns to keep.
    :param mask: boolean mask corresponding to indices. Computed if not provided.
    :return gathered_a: SparseTensor of the gathered input.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    in_size = _square_size(a.dense_shape)
    out_size = tf.size(indices)
    if mask is None:
        mask = ops.indices_to_mask(indices, in_size)
    inverse_map = _indices_to_inverse_map(indices, in_size)
    return _boolean_mask_sparse_square(
        a, mask, inverse_map=inverse_map, out_size=out_size
    )
