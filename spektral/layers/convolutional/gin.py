import tensorflow as tf
from tensorflow.keras import activations, backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from spektral.layers import ops
from spektral.layers.convolutional.gcn import GraphConv


class GINConv(GraphConv):
    r"""
    A Graph Isomorphism Network (GIN) as presented by
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826).

    **Mode**: single.

    **This layer expects sparse inputs.**

    This layer computes for each node \(i\):
    $$
        \Z_i = \textrm{MLP}\big( (1 + \epsilon) \cdot \X_i + \sum\limits_{j \in \mathcal{N}(i)} \X_j \big)
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Binary adjacency matrix of shape `([batch], N, N)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `epsilon`: unnamed parameter, see
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826), and the equation above.
    This parameter can be learned by setting `epsilon=None`, or it can be set
    to a constant value, which is what happens by default (0). In practice, it
    is safe to leave it to 0.
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 channels,
                 epsilon=None,
                 mlp_hidden=None,
                 mlp_activation='relu',
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
        self.epsilon = epsilon
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)

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
            mlp_layers.append(Dense(channels, self.mlp_activation, **layer_kwargs))
        mlp_layers.append(
            Dense(self.channels, self.activation, **layer_kwargs)
        )
        self.mlp = Sequential(mlp_layers)

        # Parameter for propagating features
        if self.epsilon is None:
            self.eps = self.add_weight(shape=(1,),
                                       initializer=self.bias_initializer,
                                       name='eps')
        else:
            # If epsilon is given, keep it constant
            self.eps = K.constant(self.epsilon)

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Enforce sparse representation
        if not K.is_sparse(fltr):
            fltr = ops.dense_to_sparse(fltr)

        # Propagation
        targets = fltr.indices[:, -2]
        sources = fltr.indices[:, -1]
        messages = tf.gather(features, sources)
        aggregated = ops.scatter_sum(targets, messages, N=tf.shape(features)[0])
        hidden = (1.0 + self.eps) * features + aggregated

        # MLP
        output = self.mlp(hidden)

        return output

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': self.mlp_activation
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess(A):
        return A
