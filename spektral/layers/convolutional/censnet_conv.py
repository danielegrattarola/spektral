import tensorflow as tf

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils.convolution import gcn_filter, incidence_matrix, line_graph


class CensNetConv(Conv):
    r"""
    A CensNet convolutional layer from the paper

    > [Co-embedding of Nodes and Edges with Graph Neural Networks](https://arxiv.org/abs/2010.13242)<br>
    > Xiaodong Jiang et al.

    This implements both the node and edge propagation rules as a single layer.

    **Mode**: single, disjoint, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - A tuple containing:
        - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be
            computed with `spektral.utils.convolution.gcn_filter`.
        - Modified line graph Laplacian of shape `([batch], n_edges, n_edges)`;
            can be computed with `spektral.utils.convolution.line_graph` and
            `spektral.utils.convolution.gcn_filter`.
        - Incidence matrix of shape `([batch], n_nodes, n_edges)`; can be
            computed with `spektral.utils.convolution.incidence_matrix`.
    - Edge features of shape `([batch], n_edges, n_edge_features)`;

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `node_channels`.
    - Edge features with the same shape as the input, but with the last
    dimension changed to `edge_channels`.

    **Arguments**

    - `node_channels`: number of output channels for the node features;
    - `edge_channels`: number of output channels for the edge features;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `node_initializer`: initializer for the node feature weights (P_n);
    - `edge_initializer`: initializer for the edge feature weights (P_e);
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `edge_regularizer`: regularization applied to the edge feature weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `edge_constraint`: constraint applied to the edge feature weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        node_channels,
        edge_channels,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        node_initializer="glorot_uniform",
        edge_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        node_regularizer=None,
        edge_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        node_constraint=None,
        edge_constraint=None,
        bias_constraint=None,
        **kwargs,
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
            **kwargs,
        )

        self.node_channels = node_channels
        self.edge_channels = edge_channels

        self.__node_initializer = tf.keras.initializers.get(node_initializer)
        self.__node_regularizer = tf.keras.regularizers.get(node_regularizer)
        self.__node_constraint = tf.keras.constraints.get(node_constraint)

        self.__edge_initializer = tf.keras.initializers.get(edge_initializer)
        self.__edge_regularizer = tf.keras.regularizers.get(edge_regularizer)
        self.__edge_constraint = tf.keras.constraints.get(edge_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_features_shape, _, edge_features_shape = input_shape
        num_input_node_features = node_features_shape[-1]
        num_input_edge_features = edge_features_shape[-1]

        self.node_kernel = self.add_weight(
            shape=(num_input_node_features, self.node_channels),
            initializer=self.kernel_initializer,
            name="node_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.edge_kernel = self.add_weight(
            shape=(num_input_edge_features, self.edge_channels),
            initializer=self.kernel_initializer,
            name="edge_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # Add separate weight vectors for the edge features for the node and
        # edge feature vectors. (These are P_n and P_e in the paper.)
        self.node_weights = self.add_weight(
            shape=(num_input_node_features, 1),
            initializer=self.__node_initializer,
            name="node_weights",
            regularizer=self.__node_regularizer,
            constraint=self.__node_constraint,
        )
        self.edge_weights = self.add_weight(
            shape=(num_input_edge_features, 1),
            initializer=self.__edge_initializer,
            name="edge_weights",
            regularizer=self.__edge_regularizer,
            constraint=self.__edge_constraint,
        )

        if self.use_bias:
            self.node_bias = self.add_weight(
                shape=(self.node_channels,),
                initializer=self.bias_initializer,
                name="node_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
            self.edge_bias = self.add_weight(
                shape=(self.edge_channels,),
                initializer=self.bias_initializer,
                name="edge_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def _bias_and_activation(self, pre_activation, *, bias_weights, mask=None):
        """
        Applies the bias, activation, and mask, if necessary.

        :param pre_activation: The layer output, pre-activation.
        :param bias_weights: The weights to use for the bias.
        :param mask: The mask to use.
        :return: The biased, activated, and masked output.
        """

        if self.use_bias:
            # Apply the bias if needed.
            pre_activation = tf.nn.bias_add(pre_activation, bias_weights)
        if mask is not None:
            pre_activation *= mask[0]
        return self.activation(pre_activation)

    def _propagate_nodes(self, inputs, mask=None):
        """
        Performs the node feature propagation step.

        :param inputs: All the inputs to the layer.
        :param mask: The mask to use.
        :return: The propagated node features.
        """
        node_features, (laplacian, _, incidence), edge_features = inputs

        weighted_edge_features = tf.matmul(edge_features, self.edge_weights)
        # Remove the extra 1-dimension.
        weighted_edge_features = tf.squeeze(weighted_edge_features, axis=[-1])
        weighted_edge_features = tf.linalg.diag(weighted_edge_features)
        weighted_edge_features = ops.modal_dot(incidence, weighted_edge_features)
        weighted_edge_features = ops.modal_dot(
            weighted_edge_features, incidence, transpose_b=True
        )

        node_adjacency = weighted_edge_features * laplacian
        output = ops.modal_dot(node_adjacency, node_features)
        output = ops.modal_dot(output, self.node_kernel)

        return self._bias_and_activation(output, bias_weights=self.node_bias, mask=mask)

    def _propagate_edges(self, inputs, mask=None):
        """
        Performs the edge feature propagation step.

        :param inputs: All the inputs to the layer.
        :param mask: The mask to use.
        :return: The propagated edge features.
        """
        node_features, (_, laplacian, incidence), edge_features = inputs

        weighted_node_features = tf.matmul(node_features, self.node_weights)
        # Remove the extra 1-dimension.
        weighted_node_features = tf.squeeze(weighted_node_features, axis=[-1])
        weighted_node_features = tf.linalg.diag(weighted_node_features)
        weighted_node_features = ops.modal_dot(
            incidence, weighted_node_features, transpose_a=True
        )
        weighted_node_features = ops.modal_dot(weighted_node_features, incidence)

        edge_adjacency = weighted_node_features * laplacian
        output = ops.modal_dot(edge_adjacency, edge_features)
        output = ops.modal_dot(output, self.edge_kernel)

        return self._bias_and_activation(output, bias_weights=self.edge_bias, mask=mask)

    def call(self, inputs, mask=None):
        node_features = self._propagate_nodes(inputs, mask=mask)
        edge_features = self._propagate_edges(inputs, mask=mask)

        return node_features, edge_features

    @property
    def config(self):
        # Get configuration for sub-components.
        node_reg = tf.keras.regularizers.serialize(self.__node_regularizer)
        node_init = tf.keras.initializers.serialize(self.__node_initializer)
        node_constraint = tf.keras.constraints.serialize(self.__node_constraint)

        edge_reg = tf.keras.regularizers.serialize(self.__edge_regularizer)
        edge_init = tf.keras.initializers.serialize(self.__edge_initializer)
        edge_constraint = tf.keras.constraints.serialize(self.__edge_constraint)

        return dict(
            node_channels=self.node_channels,
            edge_channels=self.edge_channels,
            node_regularizer=node_reg,
            node_initializer=node_init,
            node_constraint=node_constraint,
            edge_regularizer=edge_reg,
            edge_initializer=edge_init,
            edge_constraint=edge_constraint,
        )

    @staticmethod
    def preprocess(adjacency):
        laplacian = gcn_filter(adjacency)
        incidence = incidence_matrix(adjacency)
        edge_laplacian = gcn_filter(line_graph(incidence).numpy())

        return laplacian, edge_laplacian, incidence
