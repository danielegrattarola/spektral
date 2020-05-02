import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.mp import MessagePassing


class AGNNConv(MessagePassing):
    r"""
    An Attention-based Graph Neural Network (AGNN) as presented by
    [Thekumparampil et al. (2018)](https://arxiv.org/abs/1803.03735).

    **Mode**: single.

    **This layer expects sparse inputs.**

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

    def propagate(self, X, A, E=None):
        # Prepare
        N = tf.shape(X)[0]
        X_norm = K.l2_normalize(X, axis=-1)

        # Gather
        indices = ops.sparse_add_self_loops(A.indices)
        index_i, index_j = indices[:, 0], indices[:, 1]
        X_j = tf.gather(X, index_j)
        X_norm_i = tf.gather(X_norm, index_j)
        X_norm_j = tf.gather(X_norm, index_j)

        # Propagate
        messages = self.message(X_j, X_norm_i, X_norm_j, index_i, N)
        embeddings = self.aggregate(messages, index_i, N)
        output = self.update(embeddings)
        output = self.activation(output)

        return output

    def message(self, x_j, x_norm_i, x_norm_j, index_i, N):
        alpha = self.beta * tf.reduce_sum(x_norm_i * x_norm_i, axis=-1)
        alpha = ops.unsorted_segment_softmax(alpha, index_i, N)
        alpha = alpha[:, None]

        return alpha * x_j

    def get_config(self):
        config = {
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor

        return {**base_config, **config}
