import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

modes = {
    'S': 1,    # Single         (rank(a)=2, rank(b)=2)
    'M': 2,    # Mixed          (rank(a)=2, rank(b)=3)
    'iM': 3,   # Inverted mixed (rank(a)=3, rank(b)=2)
    'B': 4,    # Batch          (rank(a)=3, rank(b)=3)
    'UNK': -1  # Unknown
}


################################################################################
# Graph-related ops
################################################################################
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
    output = (A / D) / transpose(D, perm=perm)

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
            outer_index = repeat(
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
    values = tf.reshape(D, (-1, ))
    return tf.SparseTensor(indices, values, dense_shape)


################################################################################
# Sparse utils
################################################################################
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


################################################################################
# Matrix multiplication
################################################################################
def filter_dot(fltr, features):
    """
    Wrapper for matmul_A_B, specifically used to compute the matrix multiplication
    between a graph filter and node features.
    :param fltr:
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    mode = autodetect_mode(fltr, features)
    if mode == modes['S'] or mode == modes['B']:
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
    B_T = transpose(b, (1, 2, 0))
    B_T = reshape(B_T, (s_1_, -1))
    output = dot(a, B_T)
    output = reshape(output, (s_1_, s_2_, -1))
    output = transpose(output, (2, 0, 1))

    return output


def matmul_A_B(a, b):
    """
    Computes A * B, dealing automatically with sparsity and data modes.
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    mode = autodetect_mode(a, b)
    if mode == modes['M']:
        # Mixed mode (rank(a)=2, rank(b)=3)
        output = mixed_mode_dot(a, b)
    elif mode == modes['iM']:
        # Inverted mixed (rank(a)=3, rank(b)=2)
        _, s_1_a, s_2_a = K.int_shape(a)
        _, s_1_b = K.int_shape(b)
        a_flat = reshape(a, (-1, s_2_a))
        output = dot(a_flat, b)
        output = reshape(output, (-1, s_1_a, s_1_b))
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
    mode = autodetect_mode(a, b)
    if mode == modes['S'] or mode == modes['M']:
        # Single (rank(a)=2, rank(b)=2)
        # Mixed (rank(a)=2, rank(b)=3)
        a_t = transpose(a)
    elif mode == modes['iM'] or mode == modes['B']:
        # Inverted mixed (rank(a)=3, rank(b)=2)
        # Batch (rank(a)=3, rank(b)=3)
        a_t = transpose(a, (0, 2, 1))
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
    mode = autodetect_mode(a, b)
    if mode == modes['S'] or mode == modes['iM']:
        # Single (rank(a)=2, rank(b)=2)
        # Inverted mixed (rank(a)=3, rank(b)=2)
        b_t = transpose(b)
    elif mode == modes['M'] or mode == modes['B']:
        # Mixed (rank(a)=2, rank(b)=3)
        # Batch (rank(a)=3, rank(b)=3)
        b_t = transpose(b, (0, 2, 1))
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


################################################################################
# Ops related to the modes of operation (single, mixed, batch)
################################################################################
def autodetect_mode(a, b):
    """
    Return a code identifying the mode of operation (single, mixed, inverted mixed and
    batch), given a and b. See `ops.modes` for meaning of codes.
    :param a: Tensor or SparseTensor.
    :param b: Tensor or SparseTensor.
    :return: mode of operation as an integer code.
    """
    if K.ndim(b) == 2:
        if K.ndim(a) == 2:
            return modes['S']
        elif K.ndim(a) == 3:
            return modes['iM']
    elif K.ndim(b) == 3:
        if K.ndim(a) == 2:
            return modes['M']
        elif K.ndim(a) == 3:
            return modes['B']
    return modes['UNK']


################################################################################
# Misc ops
################################################################################
def transpose(a, perm=None, name=None):
    """
    Transposes a according to perm, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(a, perm=perm, name=name)


def reshape(a, shape=None, name=None):
    """
    Reshapes a according to shape, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)


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


################################################################################
# Custom ops
################################################################################
def repeat(x, repeats):
    """
    Repeats elements of a Tensor (equivalent to np.repeat, but only for 1D
    tensors).
    :param x: rank 1 Tensor;
    :param repeats: rank 1 Tensor with same shape as x, the number of
    repetitions for each element;
    :return: rank 1 Tensor, of shape `(sum(repeats), )`.
    """
    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def segment_top_k(x, I, ratio, top_k_var):
    """
    Returns indices to get the top K values in x segment-wise, according to
    the segments defined in I. K is not fixed, but it is defined as a ratio of
    the number of elements in each segment.
    :param x: a rank 1 Tensor;
    :param I: a rank 1 Tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :param top_k_var: a tf.Variable created without shape validation (i.e.,
    `tf.Variable(0.0, validate_shape=False)`);
    :return: a rank 1 Tensor containing the indices to get the top K values of
    each segment in x.
    """
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.math.ceil(ratio * tf.cast(num_nodes, tf.float32))
    to_keep = tf.cast(to_keep, tf.int32)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(x)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    # subtract 1 to ensure that filler values do not get picked
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)
    # top_k_var is a variable with unknown shape defined in the elsewhere
    top_k_var.assign(dense_y)
    dense_y = tf.tensor_scatter_nd_update(top_k_var, index[..., None], x)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = repeat(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm
