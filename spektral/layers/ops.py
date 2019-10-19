import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras import backend as K

_modes = {
    'S': 1,    # Single (rank(A)=2, rank(B)=2)
    'M': 2,    # Mixed (rank(A)=2, rank(B)=3)
    'iM': 3,   # Inverted mixed (rank(A)=3, rank(B)=2)
    'B': 4,    # Batch (rank(A)=3, rank(B)=3)
    'UNK': -1  # Unknown
}


################################################################################
# Ops for convolutions / Laplacians
################################################################################
def filter_dot(fltr, features):
    """
    Performs the multiplication of a graph filter (N x N) with the node features,
    automatically dealing with single, mixed, and batch modes.
    :param fltr: the graph filter(s) (N x N in single and mixed mode,
    batch x N x N in batch mode).
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)


def normalize_A(A):
    """
    Computes symmetric normalization of A, dealing with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: SparseTensor of rank k.
    """
    D = degrees(A)
    D = tf.sqrt(D)[:, None] + K.epsilon()
    if K.ndim(A) == 3:
        # Batch mode
        output = (A / D) / transpose(D, perm=(0, 2, 1))
    else:
        # Single mode
        output = (A / D) / transpose(D)

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
    SparseTensor. Note that the sparse degree tensor returned by this function
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
# Scipy to tf.sparse conversion
################################################################################
def sp_matrix_to_sp_tensor_value(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensorValue
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensorValue
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return tf.SparseTensorValue(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensor
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensor
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


################################################################################
# Matrix multiplication
################################################################################
def matmul_A_B(A, B):
    """
    Computes A * B, dealing with sparsity and single/batch/mixed modes
    automatically. Mixed mode multiplication also works when A has rank 3 and
    B has rank 2. Sparse multiplication does not work with batch mode.
    :param A: Tensor or SparseTensor with rank 2 or 3.
    :param B: Tensor or SparseTensor with rank 2 or 3.
    :return:
    """
    mode = autodetect_mode(A, B)
    if mode == _modes['S']:
        # Single mode
        output = single_mode_dot(A, B)
    elif mode == _modes['M']:
        # Mixed mode
        output = mixed_mode_dot(A, B)
    elif mode == _modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(A, B)
    elif mode == _modes['B']:
        # Batch mode
        # Works only with dense tensors
        output = K.batch_dot(A, B)
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_AT_B_A(A, B):
    """
    Computes A.T * B * A, dealing with sparsity and single/batch/mixed modes
    automatically. Mixed mode multiplication also works when A has rank 3 and
    B has rank 2. Sparse multiplication does not work with batch mode.
    :param A: Tensor or SparseTensor with rank 2 or 3.
    :param B: Tensor or SparseTensor with rank 2 or 3.
    :return:
    """
    mode = autodetect_mode(A, B)
    if mode == _modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(single_mode_dot(transpose(A), B), A)
    elif mode == _modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(transpose(A), B)
        if K.is_sparse(A):
            output = transpose(
                mixed_mode_dot(transpose(A), transpose(output, (0, 2, 1))),
                (0, 2, 1)
            )
        else:
            output = K.dot(output, A)
    elif mode == _modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = mixed_mode_dot(B, A)
        output = K.batch_dot(transpose(A, (0, 2, 1)), output)
    elif mode == _modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(
            K.batch_dot(
                transpose(A, (0, 2, 1)),
                B
            ),
            A
        )
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_AT_B(A, B):
    """
    Computes A.T * B, dealing with sparsity and single/batch/mixed modes
    automatically. Mixed mode multiplication also works when A has rank 3 and
    B has rank 2. Sparse multiplication does not work with batch mode.
    :param A: Tensor or SparseTensor with rank 2 or 3.
    :param B: Tensor or SparseTensor with rank 2 or 3.
    :return:
    """
    mode = autodetect_mode(A, B)
    if mode == _modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(transpose(A), B)
    elif mode == _modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(transpose(A), B)
    elif mode == _modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(transpose(A, (0, 2, 1)), B)
    elif mode == _modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(transpose(A, (0, 2, 1)), B)
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_A_BT(A, B):
    """
    Computes A * B.T, dealing with sparsity and single/batch/mixed modes
    automatically. Mixed mode multiplication also works when A has rank 3 and
    B has rank 2. Sparse multiplication does not work with batch mode.
    :param A: Tensor or SparseTensor with rank 2 or 3.
    :param B: Tensor or SparseTensor with rank 2 or 3.
    :return:
    """
    mode = autodetect_mode(A, B)
    if mode == _modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(A, transpose(B))
    elif mode == _modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(A, transpose(B, (0, 2, 1)))
    elif mode == _modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(A, transpose(B))
    elif mode == _modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(A, transpose(B, (0, 2, 1)))
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


################################################################################
# Ops related to the modes of operation (single, mixed, batch)
################################################################################
def autodetect_mode(A, B):
    """
    Return a code identifying the mode of operation (single, mixed, batch),
    given A and B. See _modes variable for meaning of codes.
    :param A: Tensor.
    :param B: Tensor.
    :return: mode of operation.
    """
    if K.ndim(B) == 2:
        if K.ndim(A) == 2:
            return _modes['S']
        elif K.ndim(A) == 3:
            return _modes['iM']
        else:
            return _modes['UNK']
    elif K.ndim(B) == 3:
        if K.ndim(A) == 2:
            return _modes['M']
        elif K.ndim(A) == 3:
            return _modes['B']
        else:
            return _modes['UNK']
    else:
        return _modes['UNK']


def single_mode_dot(A, B):
    """
    Dot product between two rank 2 matrices. Deals automatically with either A
    or B being sparse.
    :param A: rank 2 Tensor or SparseTensor.
    :param B: rank 2 Tensor or SparseTensor.
    :return: rank 2 Tensor or SparseTensor.
    """
    a_sparse = K.is_sparse(A)
    b_sparse = K.is_sparse(B)
    if a_sparse and b_sparse:
        raise ValueError('Sparse x Sparse matmul is not implemented yet.')
    elif a_sparse:
        output = tf.sparse_tensor_dense_matmul(A, B)
    elif b_sparse:
        output = transpose(
            tf.sparse_tensor_dense_matmul(
                transpose(B), transpose(A)
            )
        )
    else:
        output = tf.matmul(A, B)

    return output


def mixed_mode_dot(A, B):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', fltr, output)`, but
    works for both dense and sparse input filters.
    :param A: rank 2 Tensor or SparseTensor.
    :param B: rank 3 Tensor or SparseTensor.
    :return: rank 3 Tensor or SparseTensor.
    """
    s_0_, s_1_, s_2_ = K.int_shape(B)
    B_T = transpose(B, (1, 2, 0))
    B_T = reshape(B_T, (s_1_, -1))
    output = single_mode_dot(A, B_T)
    output = reshape(output, (s_1_, s_2_, -1))
    output = transpose(output, (2, 0, 1))

    return output


################################################################################
# Wrappers for automatic switching between dense and sparse ops
################################################################################
def transpose(A, perm=None, name=None):
    """
    Transposes A according to perm, dealing with sparse A automatically.
    :param A: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(A):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(A, perm=perm, name=name)


def reshape(A, shape=None, name=None):
    """
    Reshapes A according to shape, dealing with sparse A automatically.
    :param A: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(A):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(A, shape=shape, name=name)


################################################################################
# Misc ops
################################################################################
def matrix_power(x, k):
    """
    Computes the k-th power of a square matrix.
    :param x: a square matrix (Tensor or SparseTensor)
    :param k: exponent
    :return: matrix of same type and dtype as the input
    """
    if K.ndim(x) != 2:
        raise ValueError('x must have rank 2.')
    sparse = K.is_sparse(x)
    if sparse:
        x_dense = tf.sparse.to_dense(x)
    else:
        x_dense = x

    x_k = x_dense
    for _ in range(k - 1):
        x_k = K.dot(x_k, x_dense)

    if sparse:
        return tf.contrib.layers.dense_to_sparse(x_k)
    else:
        return x_k


def repeat(x, repeats):
    """
    Repeats elements of a Tensor (equivalent to np.repeat, but only for 1D
    tensors).
    :param x: rank 1 tensor;
    :param repeats: rank 1 tensor with same shape as x, the number of
    repetitions for each element;
    :return: rank 1 tensor, of shape `(sum(repeats), )`.
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
    :param x: a rank 1 tensor;
    :param I: a rank 1 tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :param top_k_var: a tf.Variable created without shape validation (i.e.,
    `tf.Variable(0.0, validate_shape=False)`);
    :return: a rank 1 tensor containing the indices to get the top K values of
    each segment in x.
    """
    num_nodes = tf.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.ceil(ratio * tf.cast(num_nodes, tf.float32))
    to_keep = tf.cast(to_keep, tf.int32)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(x)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    # subtract 1 to ensure that filler values do not get picked
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)
    # top_k_var is a variable with unknown shape defined in the elsewhere
    dense_y = tf.assign(top_k_var, dense_y, validate_shape=False)
    dense_y = tf.scatter_update(dense_y, index, x)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = repeat(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm