from collections.abc import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Multiply, PReLU, ReLU
from tensorflow.python.ops import gen_sparse_ops

from spektral.layers.convolutional.conv import Conv
from spektral.layers.convolutional.message_passing import MessagePassing


class XENetConv(MessagePassing):
    r"""
    A XENet convolutional layer from the paper

      > [XENet: Using a new graph convolution to accelerate the timeline for protein design on quantum computers](https://www.biorxiv.org/content/10.1101/2021.05.05.442729v1)<br>
      > Jack B. Maguire, Daniele Grattarola, Eugene Klyshko, Vikram Khipple Mulligan, Hans Melo

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    For a version of this layer that supports batch mode, you can use
    `spektral.layers.XENetDenseConv` as a drop-in replacement.

    This layer computes for each node \(i\):
    $$
        \s_{ij} = \text{PReLU} \left( (\x_{i} \| \x_{j} \| \e_{ij} \| \e_{ji}) \W^{(s)} + \b^{(s)} \right) \\
        \s^{(\text{out})}_{i} = \sum\limits_{j \in \mathcal{N}(i)} \s_{ij} \\
        \s^{(\text{in})}_{i} = \sum\limits_{j \in \mathcal{N}(i)} \s_{ji} \\
        \x_{i}' = \sigma\left( (\x_{i} \| \s^{(\text{out})}_{i} \| \s^{(\text{in})}_{i}) \W^{(n)} + \b^{(n)} \right) \\
        \e_{ij}' = \sigma\left( \s_{ij} \W^{(e)} + \b^{(e)} \right)
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Binary adjacency matrices of shape `([batch], n_nodes, n_nodes)`;
    - Edge features of shape `(num_edges, n_edge_features)`;

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `node_channels`.
    - Edge features with the same shape of the input, but the last dimension
    changed to `edge_channels`.

    **Arguments**

    - `stack_channels`: integer or list of integers, number of channels for the hidden layers;
    - `node_channels`: integer, number of output channels for the nodes;
    - `edge_channels`: integer, number of output channels for the edges;
    - `attention`: whether to use attention when aggregating the stacks;
    - `node_activation`: activation function for nodes;
    - `edge_activation`: activation function for edges;
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
        stack_channels,
        node_channels,
        edge_channels,
        attention: bool = True,
        node_activation=None,
        edge_activation=None,
        aggregate: str = "sum",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            aggregate=aggregate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.stack_channels = stack_channels
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.attention = attention
        self.node_activation = node_activation
        self.edge_activation = edge_activation

    def build(self, input_shape):
        assert len(input_shape) == 3  # X, A, E, right?
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.stack_models = []
        self.stack_model_acts = []

        if isinstance(self.stack_channels, Iterable):
            assert len(self.stack_channels) > 0

            for count, value in enumerate(self.stack_channels):
                self.stack_models.append(Dense(value, **layer_kwargs))
                if count != len(self.stack_channels) - 1:
                    self.stack_model_acts.append(ReLU())
                else:
                    self.stack_model_acts.append(PReLU())
        else:
            self.stack_models.append(Dense(self.stack_channels, **layer_kwargs))
            self.stack_model_acts.append(PReLU())

        self.node_model = Dense(
            self.node_channels, activation=self.node_activation, **layer_kwargs
        )
        self.edge_model = Dense(
            self.edge_channels, activation=self.edge_activation, **layer_kwargs
        )

        if self.attention:
            self.incoming_att_sigmoid = Dense(1, activation="sigmoid")
            self.incoming_att_multiply = Multiply()
            self.outgoing_att_sigmoid = Dense(1, activation="sigmoid")
            self.outgoing_att_multiply = Multiply()

        self.built = True

    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        x_out, e_out = self.propagate(x, a, e)

        return x_out, e_out

    def message(self, x, e=None):
        x_i = self.get_targets(x)  # Features of self
        x_j = self.get_sources(x)  # Features of neighbours

        # Features of outgoing edges are simply the edge features
        e_ij = e

        # Features of incoming edges j -> i are obtained by transposing the edge features.
        # Since TF does not allow transposing sparse matrices with rank > 2, we instead
        # re-order a tf.range(n_edges) and use it as indices to re-order the edge
        # features.
        # The following two instructions are the sparse equivalent of
        #     tf.transpose(E, perm=(1, 0, 2))
        # where E has shape (N, N, S).
        reorder_idx = gen_sparse_ops.sparse_reorder(
            tf.stack([self.index_targets, self.index_sources], axis=-1),
            tf.range(tf.shape(e)[0]),
            (self.n_nodes, self.n_nodes),
        )[1]
        e_ji = tf.gather(e, reorder_idx)

        # Concatenate the features and feed to first MLP
        stack_ij = tf.concat(
            [x_i, x_j, e_ij, e_ji], axis=-1
        )  # Shape: (n_edges, F + F + S + S)

        for stack_conv in range(0, len(self.stack_models)):
            stack_ij = self.stack_models[stack_conv](stack_ij)
            stack_ij = self.stack_model_acts[stack_conv](stack_ij)

        return stack_ij

    def aggregate(self, messages, x=None):
        if self.attention:
            incoming_att = self.incoming_att_sigmoid(messages)
            incoming = self.incoming_att_multiply([incoming_att, messages])
            incoming = self.agg(incoming, self.index_targets, self.n_nodes)
            outgoing_att = self.outgoing_att_sigmoid(messages)
            outgoing = self.outgoing_att_multiply([outgoing_att, messages])
            outgoing = self.agg(outgoing, self.index_sources, self.n_nodes)
        else:
            incoming = self.agg(messages, self.index_targets, self.n_nodes)
            outgoing = self.agg(messages, self.index_sources, self.n_nodes)

        return tf.concat([x, incoming, outgoing], axis=-1), messages

    def update(self, embeddings):
        x_new, stack_ij = embeddings

        return self.node_model(x_new), self.edge_model(stack_ij)

    @property
    def config(self):
        return {
            "stack_channels": self.stack_channels,
            "node_channels": self.node_channels,
            "edge_channels": self.edge_channels,
            "attention": self.attention,
            "node_activation": self.node_activation,
            "edge_activation": self.edge_activation,
        }


class XENetConvBatch(Conv):
    r"""
    A batch-mode version of XENetConv.

    **Mode**: batch.

    **This layer expects a dense adjacency matrix.**
    """

    def __init__(
        self,
        stack_channels,
        node_channels,
        edge_channels,
        attention: bool = True,
        node_activation=None,
        edge_activation=None,
        aggregate: str = "sum",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            aggregate=aggregate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.stack_channels = stack_channels
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.attention = attention
        self.node_activation = node_activation
        self.edge_activation = edge_activation

    def build(self, input_shape):
        assert len(input_shape) == 3  # X, A, E, right?
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        channels = input_shape[0][-1]  # last dim of X

        self.stack_models = []
        self.stack_model_acts = []

        if isinstance(self.stack_channels, Iterable):
            assert len(self.stack_channels) > 0

            for count, value in enumerate(self.stack_channels):
                self.stack_models.append(Dense(value, **layer_kwargs))
                if count != len(self.stack_channels) - 1:
                    self.stack_model_acts.append(ReLU())
                else:
                    self.stack_model_acts.append(PReLU(shared_axes=[1, 2]))
        else:
            self.stack_models.append(Dense(self.stack_channels, **layer_kwargs))
            self.stack_model_acts.append(PReLU(shared_axes=[1, 2]))

        self.node_model = Dense(
            self.node_channels, activation=self.node_activation, **layer_kwargs
        )
        self.edge_model = Dense(
            self.edge_channels, activation=self.edge_activation, **layer_kwargs
        )

        if self.attention:
            self.incoming_att_sigmoid = Dense(1, activation="sigmoid")
            self.incoming_att_multiply = Multiply()
            self.outgoing_att_sigmoid = Dense(1, activation="sigmoid")
            self.outgoing_att_multiply = Multiply()

        self.built = True

    def call(self, inputs):
        x, a, e = inputs

        assert len(x.shape) == 3
        assert len(e.shape) == 4

        N = tf.shape(x)[1]

        xi_shape = [-1, N, 1, x.shape[2]]
        xj_shape = [-1, 1, N, x.shape[2]]

        xi = tf.reshape(x, xi_shape)  # b n 1 f
        xj = tf.reshape(x, xj_shape)  # b 1 n f

        xi = tf.repeat(xi, N, axis=2)
        xj = tf.repeat(xj, N, axis=1)

        e_transpose = tf.transpose(e, perm=[0, 2, 1, 3])

        stack = tf.concat([xi, xj, e, e_transpose], axis=-1)

        for i in range(0, len(self.stack_models)):
            stack = self.stack_models[i](stack)
            stack = self.stack_model_acts[i](stack)

        e_mask_shape = [-1, tf.shape(a)[1], tf.shape(a)[2], 1]
        e_mask = tf.reshape(a, e_mask_shape)

        # zero-out elements that aren't edges in the adjacency matrix
        stack = Multiply()([stack, e_mask])

        if self.attention:
            att1 = self.incoming_att_sigmoid(stack)
            incoming_e = self.incoming_att_multiply([stack, att1])
            incoming_e = tf.keras.backend.sum(incoming_e, axis=-2, keepdims=False)
            att2 = self.outgoing_att_sigmoid(stack)
            outgoing_e = self.outgoing_att_multiply([stack, att2])
            outgoing_e = tf.keras.backend.sum(outgoing_e, axis=-3, keepdims=False)
        else:
            incoming_e = tf.keras.backend.sum(stack, axis=-2, keepdims=False)
            outgoing_e = tf.keras.backend.sum(stack, axis=-3, keepdims=False)

        final_stack = Concatenate(axis=-1)([x, incoming_e, outgoing_e])

        x = self.node_model(final_stack)
        e = self.edge_model(stack)

        # zero-out elements that aren't edges in the adjacency matrix
        e = Multiply()([e, e_mask])
        return x, e

    @property
    def config(self):
        return {
            "stack_channels": self.stack_channels,
            "node_channels": self.node_channels,
            "edge_channels": self.edge_channels,
            "attention": self.attention,
            "node_activation": self.node_activation,
            "edge_activation": self.edge_activation,
        }
