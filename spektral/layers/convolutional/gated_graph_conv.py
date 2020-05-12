import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GRUCell

from spektral.layers.convolutional.message_passing import MessagePassing


class GatedGraphConv(MessagePassing):
    r"""
    A gated graph convolutional layer as presented by
    [Li et al. (2018)](https://arxiv.org/abs/1511.05493).

    **Mode**: single, disjoint.

    **This layer expects a sparse adjacency matrix.**

    This layer repeatedly applies a GRU cell \(L\) times to the node attributes
    $$
    \begin{align}
        & \h^{(0)}_i = \X_i \| \mathbf{0} \\
        & \m^{(l)}_i = \sum\limits_{j \in \mathcal{N}(i)} \h^{(l - 1)}_j \W \\
        & \h^{(l)}_i = \textrm{GRU} \left(\m^{(l)}_i, \h^{(l - 1)}_i \right) \\
        & \Z_i = h^{(L)}_i
    \end{align}
    $$
    where \(\textrm{GRU}\) is the GRU cell.

    **Input**

    - Node features of shape `(N, F)`; note that `F` must be smaller or equal
    than `channels`.
    - Binary adjacency matrix of shape `(N, N)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `n_layers`: integer, number of iterations with the GRU cell;
    - `activation`: activation function to use;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 channels,
                 n_layers,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.channels = self.output_dim = channels
        self.n_layers = n_layers

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][1]
        if F > self.channels:
            raise ValueError('channels ({}) must be greater than the number of '
                             'input features ({}).'.format(self.channels, F))

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_layers, self.channels, self.channels),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.rnn = GRUCell(self.channels,
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer=self.bias_initializer,
                           kernel_regularizer=self.kernel_regularizer,
                           bias_regularizer=self.bias_regularizer,
                           activity_regularizer=self.activity_regularizer,
                           kernel_constraint=self.kernel_constraint,
                           bias_constraint=self.bias_constraint,
                           use_bias=self.use_bias)
        self.built = True

    def call(self, inputs):
        X, A, E = self.get_inputs(inputs)
        F = K.int_shape(X)[-1]

        to_pad = self.channels - F
        output = tf.pad(X, [[0, 0], [0, to_pad]])
        for i in range(self.n_layers):
            m = tf.matmul(output, self.kernel[i])
            m = self.propagate(m, A)
            output = self.rnn(m, [output])[0]

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'n_layers': self.n_layers,
        }
        base_config = super().get_config()
        return {**base_config, **config}
