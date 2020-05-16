from tensorflow.keras import backend as K
import tensorflow as tf
from typing import Union
import numpy as np

SINGLE  = 1   # Single         (rank(a)=2, rank(b)=2)
MIXED   = 2   # Mixed          (rank(a)=2, rank(b)=3)
iMIXED  = 3   # Inverted mixed (rank(a)=3, rank(b)=2)
BATCH   = 4   # Batch          (rank(a)=3, rank(b)=3)
UNKNOWN = -1  # Unknown


def disjoint_signal_to_batch(X: tf.Tensor, I: tf.Tensor) -> tf.Tensor:
    """
    Given an disjoint graph signal X and its segment IDs I, this op converts
    it to a batched graph signal.

    If the graphs have different orders, then we pad the node dimension with 0
    rows until they all have the same size.

    :param tf.Tensor X: Disjoint graph signal or adjacency of shape (nodes, features).
    :param I: A rank 1 Tensor with segment IDs for X;
    :return batch: Batched version of X now with shape (batch, max_nodes, features)
    """

    # TODO: Use this op to generalise SortPool to disjoint graphs
    I = tf.cast(I, tf.int32)
    X = tf.cast(X, tf.float32)

    # Number of nodes in each graph
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)

    # Getting starting index of each graph
    start_index = tf.cumsum(num_nodes, exclusive=True)

    # Number of graphs in batch
    n_graphs = tf.shape(num_nodes)[0]

    # Size of biggest graph in batch
    max_n_nodes = tf.reduce_max(num_nodes)

    # Number of overall nodes in batch
    batch_n_nodes = tf.shape(I)[0]

    # Get feature dim
    feature_dim = tf.shape(X)[-1]

    # index of non zero rows
    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(start_index, I)) + (I * max_n_nodes)

    # initial zero batch signal of correct shape

    dense = tf.zeros((n_graphs * max_n_nodes, feature_dim))

    # empty_var is a variable with unknown shape defined in the elsewhere
    dense = tf.tensor_scatter_nd_update(dense, index[..., None], X)

    batch = tf.reshape(dense, (n_graphs, max_n_nodes, feature_dim))

    return batch


def get_graph_id(node: Union[tf.Tensor, np.array],
                 graph_sizes: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
    """
    Given a node (index) and the vector of graph sizes, this function returns the ID of the graph
    containing the node.

    :param node: tf.Tensor of shape (,)
    :param graph_sizes: tf.Tensor of shape (num_graphs,) containing the sizes of the graphs ordered
    by segment ID.
    :return: tf.Tensor of shape (,). The segment ID containing the node.
    """

    cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)

    indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0,
                                   tf.int32)

    index_of_max_min = tf.reduce_sum(indicator_if_smaller) - 1

    return index_of_max_min


def get_cum_graph_size(node: Union[tf.Tensor, np.array],
                       graph_sizes: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
    """
    Returns the number of nodes contained inside graphs with smaller segment ID then node.

    :param node: tf.Tensor of shape (,). Node ID
    :param graph_sizes: tf.Tensor of shape (num_graphs,) containing the sizes of the graphs ordered
    by segment ID.
    :return: tf.Tensor of shape (,). The number of nodes contained in graphs with smaller segment ID.
    """

    graph_id = get_graph_id(node, graph_sizes)

    return tf.cumsum(graph_sizes, exclusive=True)[graph_id]


def vectorised_get_cum_graph_size(
            nodes: Union[tf.Tensor, np.array],
            graph_sizes: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
    """
    Takes a list of node ids and graph sizes ordered by segment ID and returns the
    number of nodes contained in graphs with smaller segment ID.

    :param nodes: List of node ids of shape (nodes)
    :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the
    segment IDs).
    :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs
    with smaller segment ID for each node.
    """

    def fixed_cum_graph_size(node: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
        return get_cum_graph_size(node, graph_sizes)

    return tf.map_fn(fixed_cum_graph_size, nodes)


def disjoint_adjacency_to_batch(A: tf.Tensor, I: tf.Tensor) -> tf.Tensor:
    """
    Given an disjoint adjacency A and its segment IDs I, this op converts
    it to a batched adjacency.

    If the graphs have different orders, then we pad the node dimension with 0
    rows until they all have the same size.

    :param tf.Tensor A: Disjoint graph sparse adjacency of shape (nodes, nodes).
    :param I: A rank 1 Tensor with segment IDs for X;
    :return batch: Batched version of A now with shape (batch, max_nodes, max_nodes)
    """
    I = tf.cast(I, tf.int64)
    A = tf.cast(A, tf.float32)
    indices = A.indices
    values = tf.cast(A.values, tf.int64)
    i_nodes, j_nodes = indices[:, 0], indices[:, 1]

    # Number of nodes in each graph
    graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)

    # Size of biggest graph in batch
    max_n_nodes = tf.reduce_max(graph_sizes)

    # Number of graphs in batch
    n_graphs = tf.shape(graph_sizes)[0]

    # j nodes projected to be within the range of 0 to max_n_nodes relative to their
    # connected component
    relative_j_nodes = j_nodes - vectorised_get_cum_graph_size(j_nodes, graph_sizes)

    # i nodes projected to be within the range of 0 to max_n_nodes relative to their
    # connected component
    relative_i_nodes = i_nodes - vectorised_get_cum_graph_size(i_nodes, graph_sizes)

    # Now put the i nodes in the right place relative to previous graphs who now all
    # have an order of max_n_nodes
    spaced_i_nodes = I*max_n_nodes + relative_i_nodes

    # Zip the nodes together
    new_indices = tf.transpose(tf.stack([spaced_i_nodes, relative_j_nodes]))

    # casting to same format
    new_indices = tf.cast(new_indices, tf.int32)
    n_graphs = tf.cast(n_graphs, tf.int32)
    max_n_nodes = tf.cast(max_n_nodes, tf.int32)

    # Fill in the values into a newly shaped batched adjacency
    dense_adjacency = tf.scatter_nd(new_indices, values,
                                    (n_graphs * max_n_nodes, max_n_nodes))

    # reshape to be batched
    batch = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))

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