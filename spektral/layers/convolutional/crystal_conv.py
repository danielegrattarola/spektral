from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers.convolutional.message_passing import MessagePassing


class CrystalConv(MessagePassing):
    r"""
    A crystal graph convolutional layer from the paper

    > [Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties](https://arxiv.org/abs/1710.10324)<br>
    > Tian Xie and Jeffrey C. Grossman

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \x_i + \sum\limits_{j \in \mathcal{N}(i)} \sigma \left( \z_{ij}
        \W^{(f)} + \b^{(f)} \right) \odot \g \left( \z_{ij} \W^{(s)} + \b^{(s)}
        \right)
    $$
    where \(\z_{ij} = \X_i \| \X_j \| \E_{ij} \), \(\sigma\) is a sigmoid
    activation, and \(g\) is the activation function (defined by the `activation`
    argument).

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.
    - Edge features of shape `(num_edges, n_edge_features)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
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
        aggregate="sum",
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
            aggregate=aggregate,
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

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.dense_f = Dense(self.channels, activation="sigmoid", **layer_kwargs)
        self.dense_s = Dense(self.channels, activation=self.activation, **layer_kwargs)

        self.built = True

    def message(self, x, e=None):
        x_i = self.get_i(x)
        x_j = self.get_j(x)

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        z = K.concatenate(to_concat, axis=-1)
        output = self.dense_s(z) * self.dense_f(z)

        return output

    def update(self, embeddings, x=None):
        return x + embeddings

    @property
    def config(self):
        return {"channels": self.channels}
