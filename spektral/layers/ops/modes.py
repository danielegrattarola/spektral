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


def _vectorised_get_cum_graph_size(nodes, graph_sizes):
    """
    Takes a list of node ids and graph sizes ordered by segment ID and returns the
    number of nodes contained in graphs with smaller segment ID.

    :param nodes: List of node ids of shape (nodes)
    :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the
    segment IDs).
    :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs
    with smaller segment ID for each node.
    """

    def get_cum_graph_size(node):
        cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
        indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
        graph_id = tf.reduce_sum(indicator_if_smaller) - 1
        return tf.cumsum(graph_sizes, exclusive=True)[graph_id]

    return tf.map_fn(get_cum_graph_size, nodes)


def disjoint_adjacency_to_batch(A, I):
    """
    Converts a disjoint adjacency matrix to batch node by zero-padding.

    :param A: Tensor, binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    :param I: Tensor, graph IDs of shape `(n_nodes, )`;
    :return: Tensor, batched adjacency matrix of shape `(batch, N_max, N_max)`;
    """
    I = tf.cast(I, tf.int64)
    A = tf.cast(A, tf.float32)
    indices = A.indices
    values = tf.cast(A.values, tf.int64)
    i_nodes, j_nodes = indices[:, 0], indices[:, 1]

    graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
    max_n_nodes = tf.reduce_max(graph_sizes)
    n_graphs = tf.shape(graph_sizes)[0]
    relative_j_nodes = j_nodes - _vectorised_get_cum_graph_size(j_nodes, graph_sizes)
    relative_i_nodes = i_nodes - _vectorised_get_cum_graph_size(i_nodes, graph_sizes)
    spaced_i_nodes = I * max_n_nodes + relative_i_nodes
    new_indices = tf.transpose(tf.stack([spaced_i_nodes, relative_j_nodes]))

    new_indices = tf.cast(new_indices, tf.int32)
    n_graphs = tf.cast(n_graphs, tf.int32)
    max_n_nodes = tf.cast(max_n_nodes, tf.int32)

    dense_adjacency = tf.scatter_nd(
        new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
    )
    batch = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))
    batch = tf.cast(batch, tf.float32)
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
