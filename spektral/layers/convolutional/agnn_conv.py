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

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`.

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
        X, A, E = self.get_inputs(inputs)
        X_norm = K.l2_normalize(X, axis=-1)
        output = self.propagate(X, A, E, X_norm=X_norm)
        output = self.activation(output)

        return output

    def message(self, X, X_norm=None):
        X_j = self.get_j(X)
        X_norm_i = self.get_i(X_norm)
        X_norm_j = self.get_j(X_norm)
        alpha = self.beta * tf.reduce_sum(X_norm_i * X_norm_j, axis=-1)
        alpha = ops.unsorted_segment_softmax(alpha, self.index_i, self.N)
        alpha = alpha[:, None]

        return alpha * X_j

    def get_config(self):
        config = {
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor

        return {**base_config, **config}
