from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers.convolutional.message_passing import MessagePassing


class CrystalConv(MessagePassing):
    r"""
    A Crystal Graph Convolutional layer as presented by
    [Xie & Grossman (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301).

    **Mode**: single, disjoint.

    **This layer expects a sparse adjacency matrix.**

    This layer computes for each node \(i\):
    $$
        \H_i = \X_i +
               \sum\limits_{j \in \mathcal{N}(i)}
                    \sigma \left( \z_{ij} \W^{(f)} + \b^{(f)} \right)
                    \odot
                    \g \left( \z_{ij} \W^{(s)} + \b^{(s)} \right)
    $$
    where \(\z_{ij} = \X_i \| \X_j \| \E_{ij} \), \(\sigma\) is a sigmoid
    activation, and \(g\) is the activation function (defined by the `activation`
    argument).

    **Input**

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`.
    - Edge features of shape `(num_edges, S)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
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
        super().__init__(aggregate='sum',
                         activation=activation,
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

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        self.dense_f = Dense(self.channels, activation='sigmoid', **layer_kwargs)
        self.dense_s = Dense(self.channels, activation=self.activation, **layer_kwargs)

        self.built = True

    def message(self, X, E=None):
        X_i = self.get_i(X)
        X_j = self.get_j(X)
        Z = K.concatenate((X_i, X_j, E), axis=-1)
        output = self.dense_s(Z) * self.dense_f(Z)

        return output

    def update(self, embeddings, X=None):
        return X + embeddings

    def get_config(self):
        config = {
            'channels': self.channels
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor
        return {**base_config, **config}
