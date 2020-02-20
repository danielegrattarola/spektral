import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.epsilon = epsilon
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        initializers_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        mlp_layers = []
        for i, channels in enumerate(self.mlp_hidden):
            mlp_layers.append(Dense(channels, self.mlp_activation, **initializers_kwargs))
        mlp_layers.append(
            Dense(self.channels, self.activation, **initializers_kwargs)
        )
        self.mlp = Sequential(mlp_layers)

        # Parameter for propagating features
        if self.epsilon is None:
            self.eps = self.add_weight(shape=(1,),
                                       initializer=self.bias_initializer,
                                       name='eps')
        else:
            # if epsilon is given, keep it constant
            self.eps = K.constant(self.epsilon)

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Enforce sparsity
        if not K.is_sparse(fltr):
            fltr = ops.dense_to_sparse(fltr)

        # Propagation
        features_neigh = tf.math.segment_sum(tf.gather(features, fltr.indices[:, -1]), fltr.indices[:, -2])
        hidden = (1.0 + self.eps) * features + features_neigh

        # MLP
        output = self.mlp(hidden)

        return output

    @staticmethod
    def preprocess(A):
        return A