import numpy as np
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


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
    :param kernel_initializer: initializer for the kernel matrix;
    :param kernel_regularizer: regularization applied to the kernel;
    :param activity_regularizer: regularization applied to the output;
    :param kernel_constraint: constraint applied to the kernel;
    """
    def __init__(self,
                 trainable_kernel=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(InnerProduct, self).__init__(**kwargs)
        self.trainable_kernel = trainable_kernel
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable_kernel:
            features_dim = input_shape[-1]
            self.kernel = self.add_weight(shape=(features_dim, features_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
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
            return input_shape[:-1] + (input_shape[-2], )

    def get_config(self, **kwargs):
        config = {}
        base_config = super(InnerProduct, self).get_config()
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
    :param activity_regularizer: regularization applied to the output;
    """
    def __init__(self,
                 input_dim_1=None,
                 activation=None,
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MinkowskiProduct, self).__init__(**kwargs)
        self.input_dim_1 = input_dim_1
        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)

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
            return input_shape[:-1] + (input_shape[-2], )

    def get_config(self, **kwargs):
        config = {}
        base_config = super(MinkowskiProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
