from keras import backend as K
from keras.backend import tf
import scipy.sparse as sp
import numpy as np


def sparse_bool_mask(x, mask, axis=0):
    # Only necessary if indices may have non-unique elements
    indices = tf.boolean_mask(tf.range(tf.shape(x)[axis]), mask)
    n_indices = tf.size(indices)
    # Get indices for the axis
    idx = x.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))  # TODO this has quadratic cost
    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(x.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tf.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(x.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis],
                             tf.expand_dims(idx_new, 1),
                             indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([x.dense_shape[:axis],
                           [n_indices],
                           x.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)


def mixed_mode_dot(fltr, features):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', fltr, output)`, but
    works for both dense and sparse input filters.
    :param fltr: rank 2 tensor, the filter for convolution
    :param features: rank 3 tensor, the features of the input signals
    :return:
    """
    _, m_, f_ = K.int_shape(features)
    features = K.permute_dimensions(features, [1, 2, 0])
    features = K.reshape(features, (m_, -1))
    features = K.dot(fltr, features)
    features = K.reshape(features, (m_, f_, -1))
    features = K.permute_dimensions(features, [2, 0, 1])

    return features


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
    return K.tf.SparseTensorValue(
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
    return K.tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def matrix_power(x, k):
    """
    Computes the k-th power of a square matrix.
    :param x: a square matrix (Tensor or SparseTensor)
    :param k: exponent
    :return: matrix of same type and dtype as the input
    """
    if K.is_sparse(x):
        sparse = True
        x_dense = tf.sparse.to_dense(x)
    else:
        sparse = False
        x_dense = x

    x_k = x_dense
    for _ in range(k - 1):
        x_k = K.dot(x_k, x_dense)

    if sparse:
        return tf.contrib.layers.dense_to_sparse(x_k)
    else:
        return x_k


def tf_repeat_1d(x, repeats):
    """
    Repeats each value `x[i]` a number of times `repeats[i]`.
    :param x: a rank 1 tensor;
    :param repeats: a rank 1 tensor;
    :return: a rank 1 tensor, of shape `(sum(repeats), )`.
    """
    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def top_k(scores, I, ratio, top_k_var):
    """
    Returns indices to get the top K values in `scores` segment-wise, with
    segments defined by I. K is not fixed, but it is defined as a ratio of the
    number of elements in each segment.
    :param scores: a rank 1 tensor with scores;
    :param I: a rank 1 tensor with segment IDs;
    :param ratio: float, ratio of elements to keep for each segment;
    :param top_k_var: a tf.Variable without shape validation (e.g.,
    `tf.Variable(0.0, validate_shape=False)`);
    :return: a rank 1 tensor containing the indices to get the top K values of
    each segment in `scores`.
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

    y_min = tf.reduce_min(scores)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)  # subtract 1 to ensure that filler values do not get picked
    dense_y = tf.assign(top_k_var, dense_y, validate_shape=False)  # top_k_var is a variable with unknown shape defined in the elsewhere
    dense_y = tf.scatter_update(dense_y, index, scores)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = tf_repeat_1d(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm
