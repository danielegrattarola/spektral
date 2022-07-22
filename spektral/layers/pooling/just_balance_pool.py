import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class JustBalancePool(SRCPool):
    r"""
    The Just Balance pooling layer from the paper

    > [Simplifying Clustering with Graph Neural Networks](https://arxiv.org/abs/2207.08779)<br>
    > Filippo Maria Bianchi

    **Mode**: single, batch.

    This layer learns a soft clustering of the input graph as follows:
    $$
    \begin{align}
        \S &= \textrm{MLP}(\X); \\
        \X' &= \S^\top \X \\
        \A' &= \S^\top \A \S; \\
    \end{align}
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron with softmax output.

    The layer adds the following auxiliary loss to the model
    $$
        L = - \mathrm{Tr}(\sqrt{ \S^\top \S })
    $$

    The layer can be used without a supervised loss to compute node clustering by
    minimizing the auxiliary loss.

    The layer is originally designed to be used in conjuction with a
    [GCNConv](https://graphneural.network/layers/convolution/#gcnconv)
    layer operating on the following connectivity matrix

    $$
        \tilde{\A} = \I - \delta (\I - \D^{-1/2} \A \D^{-1/2})
    $$

    **Input**

    - Node features of shape `(batch, n_nodes_in, n_node_features)`;
    - Connectivity matrix of shape
    `(batch, n_nodes_in, n_nodes_in)`;

    **Output**

    - Reduced node features of shape `(batch, n_nodes_out, n_node_features)`;
    - Reduced adjacency matrix of shape `(batch, n_nodes_out, n_nodes_out)`;
    - If `return_selection=True`, the selection matrix of shape
    `(batch, n_nodes_in, n_nodes_out)`.

    **Arguments**

    - `k`: number of output nodes;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden layer in
    the MLP used to compute cluster assignments (if `None`, the MLP has only one output
    layer);
    - `mlp_activation`: activation for the MLP layers;
    - `normalized_loss`: booelan, whether to normalize the auxiliary loss in [0,1];
    - `return_selection`: boolean, whether to return the selection matrix;
    - `kernel_initializer`: initializer for the weights of the MLP;
    - `bias_initializer`: initializer for the bias of the MLP;
    - `kernel_regularizer`: regularization applied to the weights of the MLP;
    - `bias_regularizer`: regularization applied to the bias of the MLP;
    - `kernel_constraint`: constraint applied to the weights of the MLP;
    - `bias_constraint`: constraint applied to the bias of the MLP;
    """

    def __init__(
        self,
        k,
        mlp_hidden=None,
        mlp_activation="relu",
        normalized_loss=False,
        return_selection=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            return_selection=return_selection,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.normalized_loss = normalized_loss

    def build(self, input_shape):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [Dense(self.k, "softmax", **layer_kwargs)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)
        return self.pool(x, a, i, mask=mask)

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        self.add_loss(self.balance_loss(s))

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)

        # Post-processing of A
        a_pool = tf.linalg.set_diag(
            a_pool, tf.zeros(K.shape(a_pool)[:-1], dtype=a_pool.dtype)
        )
        a_pool = ops.normalize_A(a_pool)

        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    def balance_loss(self, s):
        ss = ops.modal_dot(s, s, transpose_a=True)
        loss = -tf.linalg.trace(tf.math.sqrt(ss))

        if self.normalized_loss:
            n = float(tf.shape(s, out_type=tf.int32)[-2])
            c = float(tf.shape(s, out_type=tf.int32)[-1])
            loss = loss / tf.math.sqrt(n * c)
        return loss

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
            "normalized_loss": self.normalized_loss,
        }
        base_config = super().get_config()
        return {**base_config, **config}
