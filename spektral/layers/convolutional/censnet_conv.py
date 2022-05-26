from spektral.layers.convolutional.conv import Conv


class _CensNetNodeConv(Conv):
    r"""
    Node convolution layer from [CensNet](
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9224195)

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Edge features of shape `([batch], n_edges, n_edge_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be
        computed with `spektral.utils.convolution.gcn_filter`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `edge_initializer`: initializer for the edge feature weights;
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
        channels,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        edge_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        edge_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        edge_constraint=None,
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

        self.__edge_initializer = edge_initializer
        self.__edge_regularizer = edge_regularizer
        self.__edge_constraint = edge_constraint

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_features_shape, edge_features_shape, _ = input_shape
        num_input_node_features = node_features_shape[-1]
        num_input_edge_features = edge_features_shape[-1]

        self.kernel = self.add_weight(
            shape=(num_input_node_features, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        # Add a separate weight vector for the edge features.
        self.edge_weights = self.add_weight(
            shape=(num_input_edge_features,),
            initializer=self.__edge_initializer,
            constraint=self.__edge_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def __compute_edge_connection_matrix(self, adjacency):
        """
        Compute the edge connection matrix, which is labeled `T` in the paper.

        :param adjacency: The binary adjacency matrix, with shape
            ([batch], n_nodes, n_nodes)
        :return: The edge connection matrix, with shape
            ([batch], n_nodes, n_edges)
        """

    def call(self, inputs, mask=None):
        node_features, edge_features, adjacency = inputs
