import tensorflow as tf
import tensorflow.keras.layers as layers

from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter


class DiffuseFeatures(layers.Layer):
    r"""
    Utility layer calculating a single channel of the diffusional convolution.

    The procedure is based on [https://arxiv.org/abs/1707.01926](https://arxiv.org/abs/1707.01926)

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
    `([batch], n_nodes, n_nodes)`; Use DiffusionConvolution.preprocess to normalize.

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
        num_diffusion_steps,
        kernel_initializer,
        kernel_regularizer,
        kernel_constraint,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.K = num_diffusion_steps
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        # Initializing the kernel vector (R^K) (theta in paper)
        self.kernel = self.add_weight(
            shape=(self.K,),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

    def call(self, inputs):
        x, a = inputs

        # Calculate diffusion matrix: sum kernel_k * Attention_t^k
        # tf.polyval needs a list of tensors as the coeff. thus we
        # unstack kernel
        diffusion_matrix = tf.math.polyval(tf.unstack(self.kernel), a)

        # Apply it to X to get a matrix C = [C_1, ..., C_F] (n_nodes x n_node_features)
        # of diffused features
        diffused_features = tf.matmul(diffusion_matrix, x)

        # Now we add all diffused features (columns of the above matrix)
        # and apply a non linearity to obtain H:,q (eq. 3 in paper)
        H = tf.math.reduce_sum(diffused_features, axis=-1)

        # H has shape ([batch], n_nodes) but as it is the sum of columns
        # we reshape it to ([batch], n_nodes, 1)
        return tf.expand_dims(H, -1)


class DiffusionConv(Conv):
    r"""
      A diffusion convolution operator from the paper

      > [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting](https://arxiv.org/abs/1707.01926)<br>
      > Yaguang Li et al.

      **Mode**: single, disjoint, mixed, batch.

      **This layer expects a dense adjacency matrix.**

      Given a number of diffusion steps \(K\) and a row-normalized adjacency
      matrix \(\hat \A \), this layer calculates the \(q\)-th channel as:
      $$
      \mathbf{X}_{~:,~q}' = \sigma\left( \sum_{f=1}^{F} \left( \sum_{k=0}^{K-1}
      \theta_k {\hat \A}^k \right) \X_{~:,~f} \right)
      $$

      **Input**

      - Node features of shape `([batch], n_nodes, n_node_features)`;
      - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
      `([batch], n_nodes, n_nodes)`; Use `DiffusionConvolution.preprocess` to normalize.

      **Output**

      - Node features with the same shape as the input, but with the last
      dimension changed to `channels`.

      **Arguments**

      - `channels`: number of output channels;
      - `K`: number of diffusion steps.
      - `activation`: activation function \(\sigma\); (\(\tanh\) by default)
      - `kernel_initializer`: initializer for the weights;
      - `kernel_regularizer`: regularization applied to the weights;
      - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        channels,
        K=6,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )

        self.channels = channels
        self.K = K + 1

    def build(self, input_shape):
        self.filters = [
            DiffuseFeatures(
                num_diffusion_steps=self.K,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
            )
            for _ in range(self.channels)
        ]

    def apply_filters(self, x, a):
        # This will be a list of channels diffused features.
        # Each diffused feature is a (batch, n_nodes, 1) tensor.
        # Later we will concat all the features to get one
        # (batch, n_nodes, channels) diffused graph signal
        diffused_features = []

        # Iterating over all channels diffusion filters
        for diffusion in self.filters:
            diffused_feature = diffusion((x, a))
            diffused_features.append(diffused_feature)

        return tf.concat(diffused_features, -1)

    def call(self, inputs, mask=None):
        x, a = inputs
        output = self.apply_filters(x, a)

        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K - 1}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)
