import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GRUCell

from spektral.layers.convolutional.message_passing import MessagePassing


class GatedGraphConv(MessagePassing):
    r"""
    A gated graph convolutional layer from the paper

    > [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)<br>
    > Yujia Li et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes \(\x_i' = \h^{(L)}_i\) where:
    $$
    \begin{align}
        & \h^{(0)}_i = \x_i \| \mathbf{0} \\
        & \m^{(l)}_i = \sum\limits_{j \in \mathcal{N}(i)} \h^{(l - 1)}_j \W \\
        & \h^{(l)}_i = \textrm{GRU} \left(\m^{(l)}_i, \h^{(l - 1)}_i \right) \\
    \end{align}
    $$
    where \(\textrm{GRU}\) is a gated recurrent unit cell.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`; note that
    `n_node_features` must be smaller or equal than `channels`.
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `n_layers`: integer, number of iterations with the GRU cell;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        n_layers,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.n_layers = n_layers

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.n_layers, self.channels, self.channels),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.rnn = GRUCell(
            self.channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_bias=self.use_bias,
        )
        self.built = True

    def call(self, inputs):
        x, a, _ = self.get_inputs(inputs)
        F = K.int_shape(x)[-1]

        to_pad = self.channels - F
        ndims = len(tf.shape(x)) - 1
        output = tf.pad(x, [[0, 0]] * ndims + [[0, to_pad]])
        for i in range(self.n_layers):
            m = tf.matmul(output, self.kernel[i])
            m = self.propagate(m, a)
            output = self.rnn(m, [output])[0]

        output = self.activation(output)
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "n_layers": self.n_layers,
        }
