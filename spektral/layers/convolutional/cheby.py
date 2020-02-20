from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K

from spektral.layers import ops
from spektral.layers.convolutional.gcn import GraphConv
from spektral.utils import normalized_laplacian, rescale_laplacian


class ChebConv(GraphConv):
    r"""
    A Chebyshev convolutional layer as presented by
    [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).

    **Mode**: single, mixed, batch.

    This layer computes:
    $$
        \Z = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)}  + \b^{(k)},
    $$
    where \( \T^{(0)}, ..., \T^{(K - 1)} \) are Chebyshev polynomials of \(\tilde \L\)
    defined as
    $$
        \T^{(0)} = \X \\
        \T^{(1)} = \tilde \L \X \\
        \T^{(k \ge 2)} = 2 \cdot \tilde \L \T^{(k - 1)} - \T^{(k - 2)},
    $$
    where
    $$
        \tilde \L =  \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I
    $$
    is the normalized Laplacian with a rescaled spectrum.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - A list of K Chebyshev polynomials of shape
    `[([batch], N, N), ..., ([batch], N, N)]`; can be computed with
    `spektral.utils.convolution.chebyshev_filter`.

    **Output**

    - Node features with the same shape of the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: boolean, whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(self,
                 channels,
                 K=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.K = K
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(self.K, input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        laplacian = inputs[1]

        # Convolution
        T_0 = features
        output = ops.dot(T_0, self.kernel[0])

        if self.K > 1:
            T_1 = ops.filter_dot(laplacian, features)
            output += ops.dot(T_1, self.kernel[1])

        for k in range(2, self.K):
            T_2 = 2 * ops.filter_dot(laplacian, T_1) - T_0
            output += ops.dot(T_2, self.kernel[k])
            T_0, T_1 = T_1, T_2

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    @staticmethod
    def preprocess(A):
        L = normalized_laplacian(A)
        L = rescale_laplacian(L)
        return L