from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential

from spektral.layers import ops
from spektral.layers.convolutional.graph_conv import GraphConv


class APPNP(GraphConv):
    r"""
    A graph convolutional layer implementing the APPNP operator, as presented by
    [Klicpera et al. (2019)](https://arxiv.org/abs/1810.05997).

    This layer computes:
    $$
        \Z^{(0)} = \textrm{MLP}(\X); \\
        \Z^{(K)} = (1 - \alpha) \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \Z^{(K - 1)} +
                   \alpha \Z^{(0)},
    $$
    where \(\alpha\) is the _teleport_ probability and \(\textrm{MLP}\) is a
    multi-layer perceptron.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Modified Laplacian of shape `([batch], N, N)`; can be computed with
    `spektral.utils.convolution.localpooling_filter`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `alpha`: teleport probability during propagation;
    - `propagations`: number of propagation steps;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `dropout_rate`: dropout rate for Laplacian and MLP layers;
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
                 alpha=0.2,
                 propagations=1,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 dropout_rate=0.0,
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
        super().__init__(channels,
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
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.alpha = alpha
        self.propagations = propagations
        self.mlp_activation = activations.get(mlp_activation)
        self.dropout_rate = dropout_rate

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
        mlp_layers = []
        for i, channels in enumerate(self.mlp_hidden):
            mlp_layers.extend([
                Dropout(self.dropout_rate),
                Dense(channels, self.mlp_activation, **layer_kwargs)
            ])
        mlp_layers.append(
            Dense(self.channels, 'linear', **layer_kwargs)
        )
        self.mlp = Sequential(mlp_layers)
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Compute MLP hidden features
        mlp_out = self.mlp(features)

        # Propagation
        Z = mlp_out
        for k in range(self.propagations):
            Z = (1 - self.alpha) * ops.filter_dot(fltr, Z) + self.alpha * mlp_out

        if self.activation is not None:
            output = self.activation(Z)
        else:
            output = Z
        return output

    def get_config(self):
        config = {
            'alpha': self.alpha,
            'propagations': self.propagations,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': activations.serialize(self.mlp_activation),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
