from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.graph_conv import GraphConv
from spektral.utils import normalized_adjacency


class GraphConvSkip(GraphConv):
    r"""
    A simple convolutional layer with a skip connection.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \Z = \D^{-1/2} \A \D^{-1/2} \X \W_1 + \X \W_2 + \b
    $$
    where \( \A \) does not have self-loops (unlike in GraphConv).

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Normalized adjacency matrix of shape `([batch], N, N)`; can be computed
    with `spektral.utils.convolution.normalized_adjacency`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
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

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel_1 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_1',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_2 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_2',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel_1)
        output = ops.filter_dot(fltr, output)

        # Skip connection
        skip = K.dot(features, self.kernel_2)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    @staticmethod
    def preprocess(A):
        return normalized_adjacency(A)
