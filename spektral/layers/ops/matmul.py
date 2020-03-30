import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

from . import modes as modes
from . import ops as ops


def filter_dot(fltr, features):
    """
    Wrapper for matmul_A_B, specifically used to compute the matrix multiplication
    between a graph filter and node features.
    :param fltr:
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    mode = modes.autodetect_mode(fltr, features)
    if mode == modes.SINGLE or mode == modes.BATCH:
        return dot(fltr, features)
    else:
        # Mixed mode
        return mixed_mode_dot(fltr, features)


def dot(a, b, transpose_a=False, transpose_b=False):
    """
    Dot product between a and b along innermost dimensions, for a and b with
    same rank. Supports both dense and sparse multiplication (including
    sparse-sparse).
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with same rank as a.
    :param transpose_a: bool, transpose innermost two dimensions of a.
    :param transpose_b: bool, transpose innermost two dimensions of b.
    :return: Tensor or SparseTensor with rank 2 or 3.
    """
    a_is_sparse_tensor = isinstance(a, tf.SparseTensor)
    b_is_sparse_tensor = isinstance(b, tf.SparseTensor)
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
    works for both dense and sparse input filters.
    :param a: rank 2 Tensor or SparseTensor.
    :param b: rank 3 Tensor or SparseTensor.
    :return: rank 3 Tensor or SparseTensor.
    """
    s_0_, s_1_, s_2_ = K.int_shape(b)
    B_T = ops.transpose(b, (1, 2, 0))
    B_T = ops.reshape(B_T, (s_1_, -1))
    output = dot(a, B_T)
    output = ops.reshape(output, (s_1_, s_2_, -1))
    output = ops.transpose(output, (2, 0, 1))

    return output


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
        _, s_1_a, s_2_a = K.int_shape(a)
        _, s_1_b = K.int_shape(b)
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