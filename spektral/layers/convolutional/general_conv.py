import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import BatchNormalization, Dropout, PReLU

from spektral.layers.convolutional.message_passing import MessagePassing
from spektral.layers.ops import dot


class GeneralConv(MessagePassing):
    r"""
    A general convolutional layer from the paper

    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \mathrm{Agg} \left( \left\{ \mathrm{Act} \left( \mathrm{Dropout}
        \left( \mathrm{BN} \left( \x_j \W + \b \right) \right) \right),
        j \in \mathcal{N}(i) \right\} \right)
    $$

    where \( \mathrm{Agg} \) is an aggregation function for the messages,
    \( \mathrm{Act} \) is an activation function, \( \mathrm{Dropout} \)
    applies dropout to the node features, and \( \mathrm{BN} \) applies batch
    normalization to the node features.

    This layer supports the PReLU activation via the 'prelu' keyword.

    The default parameters of this layer are selected according to the best
    results obtained in the paper, and should provide a good performance on
    many node-level and graph-level tasks, without modifications.
    The defaults are as follows:

    - 256 channels
    - Batch normalization
    - No dropout
    - PReLU activation
    - Sum aggregation

    If you are uncertain about which layers to use for your GNN, this is a
    safe choice. Check out the original paper for more specific configurations.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `batch_norm`: bool, whether to use batch normalization;
    - `dropout`: float, dropout rate;
    - `aggregate`: string or callable, an aggregation function. Supported
    aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    - `activation`: activation function. This layer also supports the
    advanced activation PReLU by passing `activation='prelu'`.
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
        channels=256,
        batch_norm=True,
        dropout=0.0,
        aggregate="sum",
        activation="prelu",
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
            aggregate=aggregate,
            activation=None,
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
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        if activation == "prelu" or "prelu" in kwargs:
            self.activation = PReLU()
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.dropout = Dropout(self.dropout_rate)
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
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

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)

        # TODO: a = add_self_loops(a)

        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.propagate(x, a)

    @property
    def config(self):
        config = {
            "channels": self.channels,
        }
        if self.activation.__class__.__name__ == "PReLU":
            config["prelu"] = True

        return config
