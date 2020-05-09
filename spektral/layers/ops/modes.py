from tensorflow.keras import backend as K
import tensorflow as tf

SINGLE  = 1   # Single         (rank(a)=2, rank(b)=2)
MIXED   = 2   # Mixed          (rank(a)=2, rank(b)=3)
iMIXED  = 3   # Inverted mixed (rank(a)=3, rank(b)=2)
BATCH   = 4   # Batch          (rank(a)=3, rank(b)=3)
UNKNOWN = -1  # Unknown


def disjoint_to_batch(X: tf.Tensor, I: tf.Tensor) -> tf.Tensor:
    """
    Given an disjoint graph signal X and its segment IDs I, this op converts
    it to a batched graph signal.

    If the graphs have different orders, then we pad the node dimension with 0
    rows until they all have the same size.

    :param tf.Tensor X: Disjoint graph signal or adjacency of shape (nodes, features).
    :param I: A rank 1 Tensor with segment IDs for X;
    :return batch: Batched version of X now with shape (batch, max_nodes, features)
    """
    I = tf.cast(I, tf.int32)

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