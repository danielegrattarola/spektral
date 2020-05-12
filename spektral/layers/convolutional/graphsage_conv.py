import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.graph_conv import GraphConv


class GraphSageConv(GraphConv):
    r"""
    A GraphSAGE layer as presented by
    [Hamilton et al. (2017)](https://arxiv.org/abs/1706.02216).

    **Mode**: single, disjoint.

    This layer computes:
    $$
        \Z = \big[ \textrm{AGGREGATE}(\X) \| \X \big] \W + \b; \\
        \Z = \frac{\Z}{\|\Z\|}
    $$
    where \( \textrm{AGGREGATE} \) is a function to aggregate a node's
    neighbourhood. The supported aggregation methods are: sum, mean,
    max, min, and product.

    **Input**

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `aggregate_op`: str, aggregation method to use (`'sum'`, `'mean'`,
    `'max'`, `'min'`, `'prod'`);
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
                 aggregate_op='mean',
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
        if aggregate_op == 'sum':
            self.aggregate_op = ops.scatter_sum
        elif aggregate_op == 'mean':
            self.aggregate_op = ops.scatter_mean
        elif aggregate_op == 'max':
            self.aggregate_op = ops.scatter_max
        elif aggregate_op == 'min':
            self.aggregate_op = ops.scatter_sum
        elif aggregate_op == 'prod':
            self.aggregate_op = ops.scatter_prod
        elif callable(aggregate_op):
            self.aggregate_op = aggregate_op
        else:
            raise ValueError('Possbile aggragation methods: sum, mean, max, min, '
                             'prod')

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(2 * input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
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

        # Enforce sparse representation
        if not K.is_sparse(fltr):
            fltr = ops.dense_to_sparse(fltr)

        # Propagation
        indices = fltr.indices
        N = tf.shape(features, out_type=indices.dtype)[0]
        indices = ops.sparse_add_self_loops(indices, N)
        targets, sources = indices[:, -2], indices[:, -1]
        messages = tf.gather(features, sources)
        aggregated = self.aggregate_op(messages, targets, N)
        output = K.concatenate([features, aggregated])
        output = ops.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = K.l2_normalize(output, axis=-1)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'aggregate_op': self.aggregate_op
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess(A):
        return A
