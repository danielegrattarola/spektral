import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class TopKPool(SRCPool):
    r"""
    A gPool/Top-K layer from the papers

    > [Graph U-Nets](https://arxiv.org/abs/1905.05178)<br>
    > Hongyang Gao and Shuiwang Ji

    and

    > [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)<br>
    > Cătălina Cangea et al.

    **Mode**: single, disjoint.

    This layer computes:
    $$
        \y = \frac{\X\p}{\|\p\|}; \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$
    where \(\textrm{rank}(\y, K)\) returns the indices of the top K values of
    \(\y\), and \(\p\) is a learnable parameter vector of size \(F\).

    \(K\) is defined for each graph as a fraction of the number of nodes,
    controlled by the `ratio` argument.

    The gating operation \(\textrm{tanh}(\y)\) (Cangea et al.) can be replaced with a
    sigmoid (Gao & Ji).

    **Input**

    - Node features of shape `(n_nodes_in, n_node_features)`;
    - Adjacency matrix of shape `(n_nodes_in, n_nodes_in)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * n_nodes_in, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes_in, ratio * n_nodes_in)`;
    - Reduced graph IDs of shape `(ratio * n_nodes_in, )` (only in disjoint mode);
    - If `return_selection=True`, the selection mask of shape `(ratio * n_nodes_in, )`.
    - If `return_score=True`, the scoring vector of shape `(n_nodes_in, )`

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_selection`: boolean, whether to return the selection mask;
    - `return_score`: boolean, whether to return the node scoring vector;
    - `sigmoid_gating`: boolean, use a sigmoid activation for gating instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_selection=False,
        return_score=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            return_selection=return_selection,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.ratio = ratio
        self.return_score = return_score
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh

    def build(self, input_shape):
        self.n_nodes = input_shape[0][0]
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], 1),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x, a, i = self.get_inputs(inputs)
        y = K.dot(x, K.l2_normalize(self.kernel))
        output = self.pool(x, a, i, y=y)
        if self.return_score:
            output.append(y)

        return output

    def select(self, x, a, i, y=None):
        if i is None:
            i = tf.zeros(self.n_nodes)
        s = segment_top_k(y[:, 0], i, self.ratio)

        return tf.sort(s)

    def reduce(self, x, s, y=None):
        x_pool = tf.gather(x * self.gating_op(y), s)

        return x_pool

    def get_outputs(self, x_pool, a_pool, i_pool, s):
        output = [x_pool, a_pool]
        if i_pool is not None:
            output.append(i_pool)
        if self.return_selection:
            # Convert sparse indices to boolean mask
            s = tf.scatter_nd(s[:, None], tf.ones_like(s), (self.n_nodes,))
            output.append(s)

        return output

    def get_config(self):
        config = {
            "ratio": self.ratio,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def segment_top_k(x, i, ratio):
    """
    Returns indices to get the top K values in x segment-wise, according to
    the segments defined in I. K is not fixed, but it is defined as a ratio of
    the number of elements in each segment.
    :param x: a rank 1 Tensor;
    :param i: a rank 1 Tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :return: a rank 1 Tensor containing the indices to get the top K values of
    each segment in x.
    """
    i = tf.cast(i, tf.int32)
    n = tf.shape(i)[0]
    n_nodes = tf.math.segment_sum(tf.ones_like(i), i)
    batch_size = tf.shape(n_nodes)[0]
    n_nodes_max = tf.reduce_max(n_nodes)
    cumulative_n_nodes = tf.concat(
        (tf.zeros(1, dtype=n_nodes.dtype), tf.cumsum(n_nodes)[:-1]), 0
    )
    index = tf.range(n)
    index = (index - tf.gather(cumulative_n_nodes, i)) + (i * n_nodes_max)

    dense_x = tf.zeros(batch_size * n_nodes_max, dtype=x.dtype) - 1e20
    dense_x = tf.tensor_scatter_nd_update(dense_x, index[:, None], x)
    dense_x = tf.reshape(dense_x, (batch_size, n_nodes_max))

    perm = tf.argsort(dense_x, direction="DESCENDING")
    perm = perm + cumulative_n_nodes[:, None]
    perm = tf.reshape(perm, (-1,))

    k = tf.cast(tf.math.ceil(ratio * tf.cast(n_nodes, tf.float32)), i.dtype)

    # This costs more memory
    # to_rep = tf.tile(tf.constant([1., 0.]), (batch_size,))
    # rep_times = tf.reshape(tf.concat((k[:, None], (n_nodes_max - k)[:, None]), -1), (-1,))
    # mask = ops.repeat(to_rep, rep_times)
    # perm = tf.boolean_mask(perm, mask)

    # This is slower
    r_range = tf.ragged.range(k).flat_values
    r_delta = ops.repeat(tf.range(batch_size) * n_nodes_max, k)
    mask = r_range + r_delta
    perm = tf.gather(perm, mask)

    return perm
