import tensorflow as tf
from tensorflow.keras import backend as K

from . import ops as ops


def normalize_A(A):
    """
    Computes symmetric normalization of A, dealing with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: Tensor or SparseTensor of rank k.
    """
    D = degrees(A)
    D = tf.sqrt(D)[:, None] + K.epsilon()
    perm = (0, 2, 1) if K.ndim(A) == 3 else (1, 0)
    output = (A / D) / ops.transpose(D, perm=perm)

    return output


def degrees(A):
    """
    Computes the degrees of each node in A, dealing with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: Tensor or SparseTensor of rank k - 1.
    """
    if K.is_sparse(A):
        D = tf.sparse.reduce_sum(A, axis=-1)
    else:
        D = tf.reduce_sum(A, axis=-1)

    return D


def degree_matrix(A, return_sparse_batch=False):
    """
    Computes the degree matrix of A, deals with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :param return_sparse_batch: if operating in batch mode, return a
    SparseTensor. Note that the sparse degree Tensor returned by this function
    cannot be used for sparse matrix multiplication afterwards.
    :return: SparseTensor of rank k.
    """
    D = degrees(A)

    batch_mode = K.ndim(D) == 2
    N = tf.shape(D)[-1]
    batch_size = tf.shape(D)[0] if batch_mode else 1

    inner_index = tf.tile(tf.stack([tf.range(N)] * 2, axis=1), (batch_size, 1))
    if batch_mode:
        if return_sparse_batch:
            outer_index = ops.repeat(
                tf.range(batch_size), tf.ones(batch_size) * tf.cast(N, tf.float32)
            )
            indices = tf.concat([outer_index[:, None], inner_index], 1)
            dense_shape = (batch_size, N, N)
        else:
            return tf.linalg.diag(D)
    else:
        indices = inner_index
        dense_shape = (N, N)

    indices = tf.cast(indices, tf.int64)
    values = tf.reshape(D, (-1,))
    return tf.SparseTensor(indices, values, dense_shape)
