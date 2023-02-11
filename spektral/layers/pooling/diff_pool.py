import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class DiffPool(SRCPool):
    r"""
    A DiffPool layer from the paper

    > [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)<br>
    > Rex Ying et al.

    **Mode**: single, batch.

    This layer learns a soft clustering of the input graph as follows:
    $$
        \begin{align}
            \Z &= \textrm{GNN}_{embed}(\A, \X); \\
            \S &= \textrm{GNN}_{pool}(\A, \X); \\
            \X' &= \S^\top \Z; \\
            \A' &= \S^\top \A \S; \\
        \end{align}
    $$
    where:
    $$
        \textrm{GNN}_{\square}(\A, \X) = \D^{-1/2} \A \D^{-1/2} \X \W_{\square}.
    $$
    The number of output channels of \(\textrm{GNN}_{embed}\) is controlled by the
    `channels` parameter.

    Two auxiliary loss terms are also added to the model: the link prediction loss
    $$
        L_{LP} = \big\| \A - \S\S^\top \big\|_F
    $$
    and the entropy loss
    $$
        L_{E} - \frac{1}{N} \sum\limits_{i = 1}^{N} \S \log (\S).
    $$

    The layer can be used without a supervised loss to compute node clustering by
    minimizing the two auxiliary losses.

    **Input**

    - Node features of shape `(batch, n_nodes_in, n_node_features)`;
    - Adjacency matrix of shape `(batch, n_nodes_in, n_nodes_in)`;

    **Output**

    - Reduced node features of shape `(batch, n_nodes_out, channels)`;
    - Reduced adjacency matrix of shape `(batch, n_nodes_out, n_nodes_out)`;
    - If `return_selection=True`, the selection matrix of shape
    `(batch, n_nodes_in, n_nodes_out)`.

    **Arguments**

    - `k`: number of output nodes;
    - `channels`: number of output channels (if `None`, the number of output channels is
    the same as the input);
    - `return_selection`: boolean, whether to return the selection matrix;
    - `activation`: activation to apply after reduction;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        k,
        channels=None,
        return_selection=False,
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs,
    ):
        super().__init__(
            return_selection=return_selection,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs,
        )
        self.k = k
        self.channels = channels
        self.activation = activations.get(activation)

    def build(self, input_shape):
        in_channels = input_shape[0][-1]
        if self.channels is None:
            self.channels = in_channels
        self.kernel_emb = self.add_weight(
            shape=(in_channels, self.channels),
            name="kernel_emb",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.kernel_pool = self.add_weight(
            shape=(in_channels, self.k),
            name="kernel_pool",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)

        # Graph filter for GNNs
        if K.is_sparse(a):
            i_n = tf.sparse.eye(self.n_nodes, dtype=a.dtype)
            a_ = tf.sparse.add(a, i_n)
        else:
            i_n = tf.eye(self.n_nodes, dtype=a.dtype)
            a_ = a + i_n
        fltr = ops.normalize_A(a_)

        output = self.pool(x, a, i, fltr=fltr, mask=mask)
        return output

    def select(self, x, a, i, fltr=None, mask=None):
        s = ops.modal_dot(fltr, K.dot(x, self.kernel_pool))
        s = activations.softmax(s, axis=-1)
        if mask is not None:
            s *= mask[0]

        # Auxiliary losses
        lp_loss = self.link_prediction_loss(a, s)
        entr_loss = self.entropy_loss(s)
        if K.ndim(x) == 3:
            lp_loss = K.mean(lp_loss)
            entr_loss = K.mean(entr_loss)
        self.add_loss(lp_loss)
        self.add_loss(entr_loss)

        return s

    def reduce(self, x, s, fltr=None):
        z = ops.modal_dot(fltr, K.dot(x, self.kernel_emb))
        z = self.activation(z)

        return ops.modal_dot(s, z, transpose_a=True)

    def connect(self, a, s, **kwargs):
        return ops.matmul_at_b_a(s, a)

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    @staticmethod
    def link_prediction_loss(a, s):
        s_gram = ops.modal_dot(s, s, transpose_b=True)
        if K.is_sparse(a):
            lp_loss = tf.sparse.add(a, -s_gram)
        else:
            lp_loss = a - s_gram
        lp_loss = tf.norm(lp_loss, axis=(-1, -2))
        return lp_loss

    @staticmethod
    def entropy_loss(s):
        entr = tf.negative(
            tf.reduce_sum(tf.multiply(s, K.log(s + K.epsilon())), axis=-1)
        )
        entr_loss = K.mean(entr, axis=-1)
        return entr_loss

    def get_config(self):
        config = {"k": self.k, "channels": self.channels}
        base_config = super().get_config()
        return {**base_config, **config}
