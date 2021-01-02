import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class ECCConv(Conv):
    r"""
      An edge-conditioned convolutional layer (ECC) from the paper

      > [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs](https://arxiv.org/abs/1704.02901)<br>
      > Martin Simonovsky and Nikos Komodakis

    **Mode**: single, disjoint, batch, mixed.

    **In single, disjoint, and mixed mode, this layer expects a sparse adjacency
    matrix. If a dense adjacency is given as input, it will be automatically
    cast to sparse, which might be expensive.**

      This layer computes:
      $$
          \x_i' = \x_{i} \W_{\textrm{root}} + \sum\limits_{j \in \mathcal{N}(i)}
          \x_{j} \textrm{MLP}(\e_{j \rightarrow i}) + \b
      $$
      where \(\textrm{MLP}\) is a multi-layer perceptron that outputs an
      edge-specific weight as a function of edge attributes.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Binary adjacency matrices of shape `([batch], n_nodes, n_nodes)`;
    - Edge features. In single mode, shape `(num_edges, n_edge_features)`; in
    batch mode, shape `(batch, n_nodes, n_nodes, n_edge_features)`.

      **Output**

      - node features with the same shape of the input, but the last dimension
      changed to `channels`.

      **Arguments**

      - `channels`: integer, number of output channels;
      - `kernel_network`: a list of integers representing the hidden neurons of
      the kernel-generating network;
      - 'root': if False, the layer will not consider the root node for computing
      the message passing (first term in equation above), but only the neighbours.
      - `activation`: activation function;
      - `use_bias`: bool, add a bias vector to the output;
      - `kernel_initializer`: initializer for the weights;
      - `bias_initializer`: initializer for the bias vector;
      - `kernel_regularizer`: regularization applied to the weights;
      - `bias_regularizer`: regularization applied to the bias vector;
      - `activity_regularizer`: regularization applied to the output;
      - `kernel_constraint`: constraint applied to the weights;
      - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(
        self,
        channels,
        kernel_network=None,
        root=True,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.kernel_network = kernel_network
        self.root = root

    def build(self, input_shape):
        F = input_shape[0][-1]
        F_ = self.channels
        self.kernel_network_layers = []
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                self.kernel_network_layers.append(
                    Dense(
                        l,
                        name="FGN_{}".format(i),
                        activation="relu",
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                    )
                )
        self.kernel_network_layers.append(Dense(F_ * F, name="FGN_out"))

        if self.root:
            self.root_kernel = self.add_weight(
                name="root_kernel",
                shape=(F, F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        else:
            self.root_kernel = None
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.channels,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a, e = inputs

        # Parameters
        N = K.shape(x)[-2]
        F = K.int_shape(x)[-1]
        F_ = self.channels

        # Filter network
        kernel_network = e
        for layer in self.kernel_network_layers:
            kernel_network = layer(kernel_network)

        # Convolution
        mode = ops.autodetect_mode(x, a)
        if mode == modes.BATCH:
            kernel = K.reshape(kernel_network, (-1, N, N, F_, F))
            output = kernel * a[..., None, None]
            output = tf.einsum("abcde,ace->abd", output, x)
        else:
            # Enforce sparse representation
            if not K.is_sparse(a):
                warnings.warn(
                    "Casting dense adjacency matrix to SparseTensor."
                    "This can be an expensive operation. "
                )
                a = ops.dense_to_sparse(a)

            target_shape = (-1, F, F_)
            if mode == modes.MIXED:
                target_shape = (tf.shape(x)[0],) + target_shape
            kernel = tf.reshape(kernel_network, target_shape)
            index_i = a.indices[:, 1]
            index_j = a.indices[:, 0]
            messages = tf.gather(x, index_j, axis=-2)
            messages = tf.einsum("...ab,...abc->...ac", messages, kernel)
            output = ops.scatter_sum(messages, index_i, N)

        if self.root:
            output += K.dot(x, self.root_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "kernel_network": self.kernel_network,
            "root": self.root,
        }
