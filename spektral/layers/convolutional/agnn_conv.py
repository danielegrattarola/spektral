import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.message_passing import MessagePassing


class AGNNConv(MessagePassing):
    r"""
    An Attention-based Graph Neural Network (AGNN) as presented by
    [Thekumparampil et al. (2018)](https://arxiv.org/abs/1803.03735).

    **Mode**: single, disjoint.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \Z = \P\X
    $$
    where
    $$
        \P_{ij} = \frac{
            \exp \left( \beta \cos \left( \X_i, \X_j \right) \right)
        }{
            \sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}}
            \exp \left( \beta \cos \left( \X_i, \X_k \right) \right)
        }
    $$
    and \(\beta\) is a trainable parameter.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input.

    **Arguments**

    - `trainable`: boolean, if True, then beta is a trainable parameter.
    Otherwise, beta is fixed to 1;
    - `activation`: activation function to use;
    """

    def __init__(self, trainable=True, activation=None, **kwargs):
        super().__init__(aggregate='sum', activation=activation, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable:
            self.beta = self.add_weight(shape=(1,), initializer='ones', name='beta')
        else:
            self.beta = K.constant(1.)
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        x_norm = K.l2_normalize(x, axis=-1)
        output = self.propagate(x, a, x_norm=x_norm)
        output = self.activation(output)

        return output

    def message(self, x, x_norm=None):
        x_j = self.get_j(x)
        x_norm_i = self.get_i(x_norm)
        x_norm_j = self.get_j(x_norm)
        alpha = self.beta * tf.reduce_sum(x_norm_i * x_norm_j, axis=-1)
        alpha = ops.unsorted_segment_softmax(alpha, self.index_i, self.N)
        alpha = alpha[:, None]

        return alpha * x_j

    def get_config(self):
        config = {
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor

        return {**base_config, **config}
