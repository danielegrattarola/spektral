import tensorflow as tf
import tensorflow.keras.layers as layers
from spektral.layers.convolutional.graph_conv import GraphConv


class DiffuseFeatures(layers.Layer):
    r"""Utility layer calculating a single channel of the
    diffusional convolution.

    Procedure is based on https://arxiv.org/abs/1707.01926

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
    `([batch], N, N)`; Use DiffusionConvolution.preprocess to normalize.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to \(1\).

    **Arguments**

    - `num_diffusion_steps`: How many diffusion steps to consider. \(K\) in paper.
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the kernel vectors;
    - `kernel_constraint`: constraint applied to the kernel vectors;
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        kernel_initializer,
        kernel_regularizer,
        kernel_constraint,
        **kwargs
    ):
        super(DiffuseFeatures, self).__init__()

        # number of diffusino steps (K in paper)
        self.K = num_diffusion_steps

        # get regularizer, initializer and constraint for kernel
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):

        # Initializing the kernel vector (R^K)
        # (theta in paper)
        self.kernel = self.add_weight(
            shape=(self.K,),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

    def call(self, inputs):

        # Get signal X and adjacency A
        X, A = inputs

        # Calculate diffusion matrix: sum kernel_k * Attention_t^k
        # tf.polyval needs a list of tensors as the coeff. thus we
        # unstack kernel
        diffusion_matrix = tf.math.polyval(tf.unstack(self.kernel), A)

        # Apply it to X to get a matrix C = [C_1, ..., C_F] (N x F)
        # of diffused features
        diffused_features = tf.matmul(diffusion_matrix, X)

        # Now we add all diffused features (columns of the above matrix)
        # and apply a non linearity to obtain H:,q (eq. 3 in paper)
        H = tf.math.reduce_sum(diffused_features, axis=-1)

        # H has shape ([batch], N) but as it is the sum of columns
        # we reshape it to ([batch], N, 1)
        return tf.expand_dims(H, -1)


class DiffusionConv(GraphConv):
    r"""Applies Graph Diffusion Convolution as descibed by
    [Li et al. (2016)](https://arxiv.org/pdf/1707.01926.pdf)

    **Mode**: single, disjoint, mixed, batch.

    **This layer expects a dense adjacency matrix.**

    Given a number of diffusion steps \(K\) and a row normalized adjacency matrix \(\hat \A \),
    this layer calculates the q'th channel as:
    $$
    \mathbf{H}_{~:,~q} = \sigma\left(
        \sum_{f=1}^{F}
            \left(
                \sum_{k=0}^{K-1}\theta_k {\hat \A}^k
            \right)
        \X_{~:,~f}
    \right)
    $$

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
    `([batch], N, N)`; Use `DiffusionConvolution.preprocess` to normalize.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `num_diffusion_steps`: How many diffusion steps to consider. \(K\) in paper.
    - `activation`: activation function \(\sigma\); (\(\tanh\) by default)
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        channels: int,
        num_diffusion_steps: int = 6,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        kernel_constraint=None,
        activation='tanh',
        ** kwargs
    ):
        super().__init__(channels,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         kernel_constraint=kernel_constraint,
                         **kwargs)

        # number of features to generate (Q in paper)
        assert channels > 0
        self.Q = channels

        # number of diffusion steps for each output feature
        self.K = num_diffusion_steps + 1

    def build(self, input_shape):

        # We expect to receive (X, A)
        # A - Adjacency ([batch], N, N)
        # X - graph signal ([batch], N, F)
        X_shape, A_shape = input_shape

        # initialise Q diffusion convolution filters
        self.filters = []

        for _ in range(self.Q):
            layer = DiffuseFeatures(
                num_diffusion_steps=self.K,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
            )
            self.filters.append(layer)

    def apply_filters(self, X, A):
        """Applies diffusion convolution self.Q times to get a
        ([batch], N, Q) diffused graph signal

        """

        # This will be a list of Q diffused features.
        # Each diffused feature is a (batch, N, 1) tensor.
        # Later we will concat all the features to get one
        # (batch, N, Q) diffused graph signal
        diffused_features = []

        # Iterating over all Q diffusion filters
        for diffusion in self.filters:
            diffused_feature = diffusion((X, A))
            diffused_features.append(diffused_feature)

        # Concat them into ([batch], N, Q) diffused graph signal
        H = tf.concat(diffused_features, -1)

        return H

    def call(self, inputs):

        # Get graph signal X and adjacency tensor A
        X, A = inputs

        # 'single', 'batch' and 'mixed' mode are supported by
        # default, since we access the dimensions from the end
        # and everything else is broadcasted accordingly
        # if its missing.

        H = self.apply_filters(X, A)
        H = self.activation(H)

        return H
