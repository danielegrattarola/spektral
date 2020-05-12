import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Dropout

from spektral.layers import ops
from spektral.layers.convolutional.graph_conv import GraphConv
from spektral.layers.ops import modes


class GraphAttention(GraphConv):
    r"""
    A graph attention layer (GAT) as presented by
    [Velickovic et al. (2017)](https://arxiv.org/abs/1710.10903).

    **Mode**: single, disjoint, mixed, batch.

    **This layer expects dense inputs when working in batch mode.**

    This layer computes a convolution similar to `layers.GraphConv`, but
    uses the attention mechanism to weight the adjacency matrix instead of
    using the normalized Laplacian:
    $$
        \Z = \mathbf{\alpha}\X\W + \b
    $$
    where
    $$
        \mathbf{\alpha}_{ij} =
            \frac{
                \exp\left(
                    \mathrm{LeakyReLU}\left(
                        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_j]
                    \right)
                \right)
            }
            {\sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(
                    \mathrm{LeakyReLU}\left(
                        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_k]
                    \right)
                \right)
            }
    $$
    where \(\a \in \mathbb{R}^{2F'}\) is a trainable attention kernel.
    Dropout is also applied to \(\alpha\) before computing \(\Z\).
    Parallel attention heads are computed in parallel and their results are
    aggregated by concatenation or average.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Binary adjacency matrix of shape `([batch], N, N)`;

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`;
    - if `return_attn_coef=True`, a list with the attention coefficients for
    each attention head. Each attention coefficient matrix has shape
    `([batch], N, N)`.

    **Arguments**

    - `channels`: number of output channels;
    - `attn_heads`: number of attention heads to use;
    - `concat_heads`: bool, whether to concatenate the output of the attention
     heads instead of averaging;
    - `dropout_rate`: internal dropout rate for attention coefficients;
    - `return_attn_coef`: if True, return the attention coefficients for
    the given input (one N x N matrix for each head).
    - `activation`: activation function to use;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `attn_kernel_initializer`: initializer for the attention weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernels;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `attn_kernel_constraint`: constraint applied to the attention kernels;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(self,
                 channels,
                 attn_heads=1,
                 concat_heads=True,
                 dropout_rate=0.5,
                 return_attn_coef=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
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
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name='attn_kernel_neigh',
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias'
            )

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]

        mode = ops.autodetect_mode(A, X)
        if mode == modes.SINGLE and K.is_sparse(A):
            output, attn_coef = self._call_single(X, A)
        else:
            output, attn_coef = self._call_dense(X, A)

        if self.concat_heads:
            shape = output.shape[:-2] + [self.attn_heads * self.channels]
            shape = [d if d is not None else -1 for d in shape]
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias

        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, X, A):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_neighs = ops.transpose(self.attn_kernel_neighs, (2, 1, 0))

        # Prepare message-passing
        indices = A.indices
        N = tf.shape(X, out_type=indices.dtype)[0]
        indices = ops.sparse_add_self_loops(indices, N)
        targets, sources = indices[:, -2], indices[:, -1]

        # Update node features
        X = ops.dot(X, kernel)
        X = tf.reshape(X, (-1, self.attn_heads, self.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(X * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)
        attn_for_neighs = tf.reduce_sum(X * attn_kernel_neighs, -1)
        attn_for_neighs = tf.gather(attn_for_neighs, sources)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)
        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]

        # Update representation
        output = attn_coef * tf.gather(X, sources)
        output = ops.scatter_sum(output, targets, N)

        return output, attn_coef

    def _call_dense(self, X, A):
        shape = tf.shape(A)[:-1]
        A = tf.linalg.set_diag(A, tf.zeros(shape, A.dtype))
        A = tf.linalg.set_diag(A, tf.ones(shape, A.dtype))
        X = tf.einsum("...NI , IHO -> ...NHO", X, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", X, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", X, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = -10e9 * (1.0 - A)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, X)

        return output, attn_coef

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.output_dim,)
        return output_shape

    def get_config(self):
        config = {
            'attn_heads': self.attn_heads,
            'concat_heads': self.concat_heads,
            'dropout_rate': self.dropout_rate,
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess(A):
        return A
