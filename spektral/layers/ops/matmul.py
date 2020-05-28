import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

from . import modes as modes
from . import ops as ops


def dot(a, b, transpose_a=False, transpose_b=False):
    """
    Dot product between `a` and `b`, with automatic handling of batch dimensions.
    Supports both dense and sparse multiplication (including sparse-sparse).
    The innermost dimension of `a` must match the outermost dimension of `b`,
    unless there is a shared batch dimension.
    Note that doing sparse-sparse multiplication of any rank and sparse-dense
    multiplication with rank higher than 2 may result in slower computations.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :param transpose_a: bool, transpose innermost two dimensions of a.
    :param transpose_b: bool, transpose innermost two dimensions of b.
    :return: Tensor or SparseTensor with rank 2 or 3.
    """
    a_is_sparse_tensor = isinstance(a, tf.SparseTensor)
    b_is_sparse_tensor = isinstance(b, tf.SparseTensor)

    # Handle case where we can use faster sparse-dense matmul
    if K.ndim(a) == 2 and K.ndim(b) == 2:
        if transpose_a:
            a = ops.transpose(a)
        if transpose_b:
            b = ops.transpose(b)
        if a_is_sparse_tensor and not b_is_sparse_tensor:
            return tf.sparse.sparse_dense_matmul(a, b)
        elif not a_is_sparse_tensor and b_is_sparse_tensor:
            return ops.transpose(
                tf.sparse.sparse_dense_matmul(ops.transpose(b), ops.transpose(a))
            )

    # Fallthrough to tfsp implementation
    # Defaults to tf.matmul if neither is sparse
    if a_is_sparse_tensor:
        a = tfsp.CSRSparseMatrix(a)
    if b_is_sparse_tensor:
        b = tfsp.CSRSparseMatrix(b)
    out = tfsp.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    if hasattr(out, 'to_sparse_tensor'):
        return out.to_sparse_tensor()

    return out


def mixed_mode_dot(a, b):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', a, b)`, but
    works for both dense and sparse inputs.
    :param a: Tensor or SparseTensor with rank 2.
    :param b: Tensor or SparseTensor with rank 3.
    :return: Tensor or SparseTensor with rank 3.
    """
    s_0_, s_1_, s_2_ = K.int_shape(b)
    B_T = ops.transpose(b, (1, 2, 0))
    B_T = ops.reshape(B_T, (s_1_, -1))
    output = dot(a, B_T)
    output = ops.reshape(output, (s_1_, s_2_, -1))
    output = ops.transpose(output, (2, 0, 1))

    return output


def filter_dot(fltr, features):
    """
    Computes the matrix multiplication between a graph filter and node features,
    automatically handling data modes.
    :param fltr: Tensor or SparseTensor of rank 2 or 3.
    :param features: Tensor or SparseTensor of rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    mode = modes.autodetect_mode(fltr, features)
    if mode == modes.SINGLE or mode == modes.BATCH:
        return dot(fltr, features)
    else:
        return mixed_mode_dot(fltr, features)


def matmul_A_B(a, b):
    """
    Computes A * B, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    mode = modes.autodetect_mode(a, b)
    if mode == modes.MIXED:
        # Mixed mode (rank(a)=2, rank(b)=3)
        output = mixed_mode_dot(a, b)
    elif mode == modes.iMIXED:
        # Inverted mixed (rank(a)=3, rank(b)=2)
        # This implementation is faster than using rank 3 sparse matmul with tfsp
        s_1_a, s_2_a = tf.shape(a)[1], tf.shape(a)[2]
        s_1_b = tf.shape(b)[1]
        a_flat = ops.reshape(a, (-1, s_2_a))
        output = dot(a_flat, b)
        output = ops.reshape(output, (-1, s_1_a, s_1_b))
    else:
        # Single (rank(a)=2, rank(b)=2) and batch (rank(a)=3, rank(b)=3) mode
        output = dot(a, b)

    return output


def matmul_AT_B(a, b):
    """
    Computes A.T * B, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    mode = modes.autodetect_mode(a, b)
    if mode == modes.SINGLE or mode == modes.MIXED:
        # Single (rank(a)=2, rank(b)=2)
        # Mixed (rank(a)=2, rank(b)=3)
        a_t = ops.transpose(a)
    elif mode == modes.iMIXED or mode == modes.BATCH:
        # Inverted mixed (rank(a)=3, rank(b)=2)
        # Batch (rank(a)=3, rank(b)=3)
        a_t = ops.transpose(a, (0, 2, 1))
    else:
        raise ValueError('Expected ranks to be 2 or 3, got {} and {}'.format(
            K.ndim(a), K.ndim(b)
        ))

    return matmul_A_B(a_t, b)


def matmul_A_BT(a, b):
    """
    Computes A * B.T, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    mode = modes.autodetect_mode(a, b)
    if mode == modes.SINGLE or mode == modes.iMIXED:
        # Single (rank(a)=2, rank(b)=2)
        # Inverted mixed (rank(a)=3, rank(b)=2)
        b_t = ops.transpose(b)
    elif mode == modes.MIXED or mode == modes.BATCH:
        # Mixed (rank(a)=2, rank(b)=3)
        # Batch (rank(a)=3, rank(b)=3)
        b_t = ops.transpose(b, (0, 2, 1))
    else:
        raise ValueError('Expected ranks to be 2 or 3, got {} and {}'.format(
            K.ndim(a), K.ndim(b)
        ))

    return matmul_A_B(a, b_t)


def matmul_AT_B_A(a, b):
    """
    Computes A.T * B * A, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    at_b = matmul_AT_B(a, b)
    at_b_a = matmul_A_B(at_b, a)

    return at_b_a


def matmul_A_B_AT(a, b):
    """
    Computes A * B * A.T, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    b_at = matmul_A_BT(a, b)
    a_b_at = matmul_A_B(a, b_at)

    return a_b_at


def matrix_power(a, k):
    """
    If a is a square matrix, computes a^k. If a is a rank 3 Tensor of square
    matrices, computes the exponent of each inner matrix.
    :param a: Tensor or SparseTensor with rank 2 or 3. The innermost two
    dimensions must be the same.
    :param k: int, the exponent to which to raise the matrices.
    :return: Tensor or SparseTensor with same rank as the input.
    """
    x_k = a
    for _ in range(k - 1):
        x_k = matmul_A_B(a, x_k)

    return x_k
