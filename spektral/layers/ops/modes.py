import tensorflow as tf
from tensorflow.keras import backend as K

SINGLE = 1    # Single         (rank(a)=2, rank(b)=2)
MIXED = 2     # Mixed          (rank(a)=2, rank(b)=3)
iMIXED = 3    # Inverted mixed (rank(a)=3, rank(b)=2)
BATCH = 4     # Batch          (rank(a)=3, rank(b)=3)
UNKNOWN = -1  # Unknown


def disjoint_signal_to_batch(X, I):
    """
    Converts a disjoint graph signal to batch node by zero-padding.

    :param X: Tensor, node features of shape (nodes, features).
    :param I: Tensor, graph IDs of shape `(N, )`;
    :return batch: Tensor, batched node features of shape (batch, N_max, F)
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

    :param A: Tensor, binary adjacency matrix of shape `(N, N)`;
    :param I: Tensor, graph IDs of shape `(N, )`;
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


def autodetect_mode(a, b):
    """
    Return a code identifying the mode of operation (single, mixed, inverted mixed and
    batch), given a and b. See `ops.modes` for meaning of codes.
    :param a: Tensor or SparseTensor.
    :param b: Tensor or SparseTensor.
    :return: mode of operation as an integer code.
    """
    a_dim = K.ndim(a)
    b_dim = K.ndim(b)
    if b_dim == 2:
        if a_dim == 2:
            return SINGLE
        elif a_dim == 3:
            return iMIXED
    elif b_dim == 3:
        if a_dim == 2:
            return MIXED
        elif a_dim == 3:
            return BATCH
    return UNKNOWN
