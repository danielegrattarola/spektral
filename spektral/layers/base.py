import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import smart_cond

from spektral.layers import ops


class SparseDropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

    Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        return tf.shape(inputs.values)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return self.sparse_dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate
            )

        output = smart_cond.smart_cond(training,
                                       dropped_inputs,
                                       lambda: inputs)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def sparse_dropout(x, rate, noise_shape=None, seed=None):
        random_tensor = tf.random.uniform(noise_shape, seed=seed, dtype=x.dtype)
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        keep_mask = random_tensor >= rate
        output = tf.sparse.retain(x, keep_mask) * scale

        return output


class InnerProduct(Layer):
    r"""
    Computes the inner product between elements of a 2d Tensor:
    $$
        \langle \x, \x \rangle = \x\x^\top.
    $$

    **Mode**: single.

    **Input**

    - Tensor of shape `(N, M)`;

    **Output**

    - Tensor of shape `(N, N)`.

    :param trainable_kernel: add a trainable square matrix between the inner
    product (e.g., `X @ W @ X.T`);
    :param activation: activation function to use;
    :param kernel_initializer: initializer for the weights;
    :param kernel_regularizer: regularization applied to the kernel;
    :param kernel_constraint: constraint applied to the kernel;
    """

    def __init__(self,
                 trainable_kernel=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.trainable_kernel = trainable_kernel
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable_kernel:
            features_dim = input_shape[-1]
            self.kernel = self.add_weight(shape=(features_dim, features_dim),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs):
        if self.trainable_kernel:
            output = K.dot(K.dot(inputs, self.kernel), K.transpose(inputs))
        else:
            output = K.dot(inputs, K.transpose(inputs))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return (None, None)
        else:
            return input_shape[:-1] + (input_shape[-2],)

    def get_config(self, **kwargs):
        config = {
            'trainable_kernel': self.trainable_kernel,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinkowskiProduct(Layer):
    r"""
    Computes the hyperbolic inner product between elements of a rank 2 Tensor:
    $$
        \langle \x, \x \rangle = \x \,
        \begin{pmatrix}
            \I_{d \times d} & 0 \\
            0              & -1
        \end{pmatrix} \, \x^\top.
    $$

    **Mode**: single.

    **Input**

    - Tensor of shape `(N, M)`;

    **Output**

    - Tensor of shape `(N, N)`.

    :param input_dim_1: first dimension of the input Tensor; set this if you
    encounter issues with shapes in your model, in order to provide an explicit
    output shape for your layer.
    :param activation: activation function to use;
    """

    def __init__(self,
                 input_dim_1=None,
                 activation=None,
                 **kwargs):

        super(MinkowskiProduct, self).__init__(**kwargs)
        self.input_dim_1 = input_dim_1
        self.activation = activations.get(activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs):
        F = K.int_shape(inputs)[-1]
        minkowski_prod_mat = np.eye(F)
        minkowski_prod_mat[-1, -1] = -1.
        minkowski_prod_mat = K.constant(minkowski_prod_mat)
        output = K.dot(inputs, minkowski_prod_mat)
        output = K.dot(output, K.transpose(inputs))
        output = K.clip(output, -10e9, -1.)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            if self.input_dim_1 is None:
                return (None, None)
            else:
                return (self.input_dim_1, self.input_dim_1)
        else:
            return input_shape[:-1] + (input_shape[-2],)

    def get_config(self, **kwargs):
        config = {
            'input_dim_1': self.input_dim_1,
            'activation': self.activation
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Disjoint2Batch(Layer):
    r"""Utility layer that converts data from disjoint mode to batch mode by
    zero-padding the node features and adjacency matrices.

    **Mode**: disjoint.

    **This layer expects a sparse adjacency matrix.**

    **Input**

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`;
    - Graph IDs of shape `(N, )`;

    **Output**

    - Batched node features of shape `(batch, N_max, F)`;
    - Batched adjacency matrix of shape `(batch, N_max, N_max)`;
    """

    def __init__(self):
        super(Disjoint2Batch, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, **kwargs):
        X, A, I = inputs

        batch_X = ops.disjoint_signal_to_batch(X, I)
        batch_A = ops.disjoint_adjacency_to_batch(A, I)

        # Ensure that the channel dimension is known
        batch_X.set_shape((None, None, X.shape[-1]))
        batch_A.set_shape((None, None, None))

        return batch_X, batch_A
