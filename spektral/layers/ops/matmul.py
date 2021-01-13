import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

from . import modes as modes
from . import ops as ops


def dot(a, b):
    """
    Computes a @ b, for a, b of the same rank (both 2 or both 3).

    If the rank is 2, then the innermost dimension of `a` must match the
    outermost dimension of `b`.
    If the rank is 3, the first dimension of `a` and `b` must be equal and the
    function computes a batch matmul.

    Supports both dense and sparse multiplication (including sparse-sparse).

    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with same rank as b.
    :return: Tensor or SparseTensor with rank 2 or 3.
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)
    assert a_ndim == b_ndim, "Expected equal ranks, got {} and {}" "".format(
        a_ndim, b_ndim
    )
    a_is_sparse = K.is_sparse(a)
    b_is_sparse = K.is_sparse(b)

    # Handle cases: rank 2 sparse-dense, rank 2 dense-sparse
    # In these cases we can use the faster sparse-dense matmul of tf.sparse
    if a_ndim == 2:
        if a_is_sparse and not b_is_sparse:
            return tf.sparse.sparse_dense_matmul(a, b)
        if not a_is_sparse and b_is_sparse:
            return ops.transpose(
                tf.sparse.sparse_dense_matmul(ops.transpose(b), ops.transpose(a))
            )

    # Handle cases: rank 2 sparse-sparse, rank 3 sparse-dense,
    # rank 3 dense-sparse, rank 3 sparse-sparse
    # In these cases we can use the tfsp.CSRSparseMatrix implementation (slower,
    # but saves memory)
    if a_is_sparse:
        a = tfsp.CSRSparseMatrix(a)
    if b_is_sparse:
        b = tfsp.CSRSparseMatrix(b)
    if a_is_sparse or b_is_sparse:
        out = tfsp.matmul(a, b)
        if hasattr(out, "to_sparse_tensor"):
            return out.to_sparse_tensor()
        else:
            return out

    # Handle case: rank 2 dense-dense, rank 3 dense-dense
    # Here we use the standard dense operation
    return tf.matmul(a, b)


def mixed_mode_dot(a, b):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', a, b)`, but
    works for both dense and sparse inputs.

    :param a: Tensor or SparseTensor with rank 2.
    :param b: Tensor or SparseTensor with rank 3.
    :return: Tensor or SparseTensor with rank 3.
    """
    shp = K.int_shape(b)
    b_t = ops.transpose(b, (1, 2, 0))
    b_t = ops.reshape(b_t, (shp[1], -1))
    output = dot(a, b_t)
    output = ops.reshape(output, (shp[1], shp[2], -1))
    output = ops.transpose(output, (2, 0, 1))

    return output


def modal_dot(a, b, transpose_a=False, transpose_b=False):
    """
    Computes the matrix multiplication of a and b, handling the data modes
    automatically.

    This is a wrapper to standard matmul operations, for a and b with rank 2
    or 3, that:

    - Supports automatic broadcasting of the "batch" dimension if the two inputs
    have different ranks.
    - Supports any combination of dense and sparse inputs.

    This op is useful for multiplying matrices that represent batches of graphs
    in the different modes, for which the adjacency matrices may or may not be
    sparse and have different ranks from the node attributes.

    Additionally, it can also support the case where we have many adjacency
    matrices and only one graph signal (which is uncommon, but may still happen).

    If you know a-priori the type and shape of the inputs, it may be faster to
    use the built-in functions of TensorFlow directly instead.

    Examples:

        - `a` rank 2, `b` rank 2 -> `a @ b`
        - `a` rank 3, `b` rank 3 -> `[a[i] @ b[i] for i in range(len(a))]`
        - `a` rank 2, `b` rank 3 -> `[a @ b[i] for i in range(len(b))]`
        - `a` rank 3, `b` rank 2 -> `[a[i] @ b for i in range(len(a))]`

    :param a: Tensor or SparseTensor with rank 2 or 3;
    :param b: Tensor or SparseTensor with rank 2 or 3;
    :param transpose_a: transpose the innermost 2 dimensions of `a`;
    :param transpose_b: transpose the innermost 2 dimensions of `b`;
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)
    assert a_ndim in (2, 3), "Expected a of rank 2 or 3, got {}".format(a_ndim)
    assert b_ndim in (2, 3), "Expected b of rank 2 or 3, got {}".format(b_ndim)

    if transpose_a:
        perm = None if a_ndim == 2 else (0, 2, 1)
        a = ops.transpose(a, perm)
    if transpose_b:
        perm = None if b_ndim == 2 else (0, 2, 1)
        b = ops.transpose(b, perm)

    if a_ndim == b_ndim:
        # ...ij,...jk->...ik
        return dot(a, b)
    elif a_ndim == 2:
        # ij,bjk->bik
        return mixed_mode_dot(a, b)
    else:  # a_ndim == 3
        # bij,jk->bik
        if not K.is_sparse(a) and not K.is_sparse(b):
            # Immediately fallback to standard dense matmul, no need to reshape
            return tf.matmul(a, b)

        # If either input is sparse, we use dot(a, b)
        # This implementation is faster than using rank 3 sparse matmul with tfsp
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)
        a_flat = ops.reshape(a, (-1, a_shape[2]))
        output = dot(a_flat, b)
        return ops.reshape(output, (-1, a_shape[1], b_shape[1]))


def matmul_at_b_a(a, b):
    """
    Computes a.T @ b @ a, for a, b with rank 2 or 3.

    Supports automatic broadcasting of the "batch" dimension if the two inputs
    have different ranks.
    Supports any combination of dense and sparse inputs.

    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    at_b = modal_dot(a, b, transpose_a=True)
    at_b_a = modal_dot(at_b, a)

    return at_b_a


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
        x_k = modal_dot(a, x_k)

    return x_k
