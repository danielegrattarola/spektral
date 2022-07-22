import tensorflow as tf
from tensorflow.keras import backend as K

SINGLE = 1  # Single mode    rank(x) = 2, rank(a) = 2
DISJOINT = SINGLE  # Disjoint mode  rank(x) = 2, rank(a) = 2
BATCH = 3  # Batch mode     rank(x) = 3, rank(a) = 3
MIXED = 4  # Mixed mode     rank(x) = 3, rank(a) = 2


def disjoint_signal_to_batch(X, I):
    """
    Converts a disjoint graph signal to batch node by zero-padding.

    :param X: Tensor, node features of shape (nodes, features).
    :param I: Tensor, graph IDs of shape `(n_nodes, )`;
    :return batch: Tensor, batched node features of shape (batch, N_max, n_node_features)
    """
    I = tf.cast(I, tf.int32)
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)
    start_index = tf.cumsum(num_nodes, exclusive=True)
    n_graphs = tf.shape(num_nodes)[0]
    max_n_nodes = tf.reduce_max(num_nodes)
    batch_n_nodes = tf.shape(I)[0]
    feature_dim = tf.shape(X)[-1]

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(start_index, I)) + (I * max_n_nodes)
    dense = tf.zeros((n_graphs * max_n_nodes, feature_dim), dtype=X.dtype)
    dense = tf.tensor_scatter_nd_update(dense, index[..., None], X)

    batch = tf.reshape(dense, (n_graphs, max_n_nodes, feature_dim))

    return batch


def disjoint_adjacency_to_batch(A, I):
    """
    Converts a disjoint adjacency matrix to batch node by zero-padding.

    :param A: Tensor, binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    :param I: Tensor, graph IDs of shape `(n_nodes, )`;
    :return: Tensor, batched adjacency matrix of shape `(batch, N_max, N_max)`;
    """
    I = tf.cast(I, tf.int64)
    indices = A.indices
    values = A.values
    i_nodes, j_nodes = indices[:, 0], indices[:, 1]

    graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
    max_n_nodes = tf.reduce_max(graph_sizes)
    n_graphs = tf.shape(graph_sizes)[0]

    offset = tf.gather(I, i_nodes)
    offset = tf.gather(tf.cumsum(graph_sizes, exclusive=True), offset)

    relative_j_nodes = j_nodes - offset
    relative_i_nodes = i_nodes - offset

    spaced_i_nodes = tf.gather(I, i_nodes) * max_n_nodes + relative_i_nodes
    new_indices = tf.transpose(tf.stack([spaced_i_nodes, relative_j_nodes]))

    n_graphs = tf.cast(n_graphs, new_indices.dtype)
    max_n_nodes = tf.cast(max_n_nodes, new_indices.dtype)

    dense_adjacency = tf.scatter_nd(
        new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
    )
    batch = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))

    return batch


def autodetect_mode(x, a):
    """
    Returns a code that identifies the data mode from the given node features
    and adjacency matrix(s).
    The output of this function can be used as follows:

    ```py
    from spektral.layers.ops import modes
    mode = modes.autodetect_mode(x, a)
    if mode == modes.SINGLE:
        print('Single!')
    elif mode == modes.BATCH:
        print('Batch!')
    elif mode == modes.MIXED:
        print('Mixed!')
    ```

    :param x: Tensor or SparseTensor representing the node features
    :param a: Tensor or SparseTensor representing the adjacency matrix(s)
    :return: mode of operation as an integer code.
    """
    x_ndim = K.ndim(x)
    a_ndim = K.ndim(a)
    if x_ndim == 2 and a_ndim == 2:
        return SINGLE
    elif x_ndim == 3 and a_ndim == 3:
        return BATCH
    elif x_ndim == 3 and a_ndim == 2:
        return MIXED
    else:
        raise ValueError(
            "Unknown mode for inputs x, a with ranks {} and {}"
            "respectively.".format(x_ndim, a_ndim)
        )
