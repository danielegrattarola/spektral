import tensorflow as tf
from spektral.layers.convolutional import gcn_conv


class GCNModel(tf.keras.Model):
    """
    Model with architecture and default hyperparameters consistent with GCN paper.

    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`
    - Weighted adjacency matrix of shape `([batch], n_nodes, n_nodes)`

    **Output**

    - Softmax predictions with shape `([batch], n_nodes, num_classes)`.

    **Arguments**

    - `num_classes`: number of channels in output;
    - `channels`: number of channels in first GCNConv layer;
    - `dropout_rate`: `rate` used in `Dropout` layers;
    - `l2_reg`: l2 regularization strength;
    - `num_input_channels`: number of input channels, required for tf 2.1;
    - `**kwargs`: passed to `Model.__init__`.
    """

    def __init__(
            self,
            num_classes,
            channels=16,
            dropout_rate=0.5,
            l2_reg=2.5e-4,
            num_input_channels=None,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.num_input_channels = num_input_channels
        reg = tf.keras.regularizers.l2(l2_reg)
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = gcn_conv.GCNConv(
            channels, activation="relu", kernel_regularizer=reg, use_bias=False
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = gcn_conv.GCNConv(num_classes, activation="softmax", use_bias=False)

        if tf.version.VERSION < '2.2':
            if num_input_channels is None:
                raise ValueError("num_input_channels required for tf < 2.2")
            x = tf.keras.Input((num_input_channels,), dtype=tf.float32)
            a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
            self._set_inputs((x, a))


    def get_config(self):
        return dict(
            num_classes=self.num_classes,
            channels=self.channels,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            num_input_channels=self.num_input_channels
        )

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Inputs should be (x, a), got {}".format(inputs))
        x, a = inputs
        if self.num_input_channels is None:
            self.num_input_channels = x.shape[-1]
        else:
            assert self.num_input_channels == x.shape[-1]
        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])

