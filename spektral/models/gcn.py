import tensorflow as tf

from spektral.layers.convolutional import gcn_conv


class GCN(tf.keras.Model):
    """
    This model, with its default hyperparameters, implements the architecture
    from the paper:

    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`
    - Weighted adjacency matrix of shape `([batch], n_nodes, n_nodes)`

    **Output**

    - Softmax predictions with shape `([batch], n_nodes, n_labels)`.

    **Arguments**

    - `n_labels`: number of channels in output;
    - `channels`: number of channels in first GCNConv layer;
    - `activation`: activation of the first GCNConv layer;
    - `output_activation`: activation of the second GCNConv layer;
    - `use_bias`: whether to add a learnable bias to the two GCNConv layers;
    - `dropout_rate`: `rate` used in `Dropout` layers;
    - `l2_reg`: l2 regularization strength;
    - `n_input_channels`: number of input channels, required for tf 2.1;
    - `**kwargs`: passed to `Model.__init__`.
    """

    def __init__(
        self,
        n_labels,
        channels=16,
        activation="relu",
        output_activation="softmax",
        use_bias=False,
        dropout_rate=0.5,
        l2_reg=2.5e-4,
        n_input_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.n_input_channels = n_input_channels
        reg = tf.keras.regularizers.l2(l2_reg)
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = gcn_conv.GCNConv(
            n_labels, activation=output_activation, use_bias=use_bias
        )

        if tf.version.VERSION < "2.2":
            if n_input_channels is None:
                raise ValueError("n_input_channels required for tf < 2.2")
            x = tf.keras.Input((n_input_channels,), dtype=tf.float32)
            a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
            self._set_inputs((x, a))

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            n_input_channels=self.n_input_channels,
        )

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Inputs should be (x, a), got {}".format(inputs))
        x, a = inputs
        if self.n_input_channels is None:
            self.n_input_channels = x.shape[-1]
        else:
            assert self.n_input_channels == x.shape[-1]
        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])
