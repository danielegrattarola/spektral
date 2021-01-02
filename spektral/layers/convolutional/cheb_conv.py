from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import normalized_laplacian, rescale_laplacian


class ChebConv(Conv):
    r"""
    A Chebyshev convolutional layer from the paper

    > [Convolutional Neural Networks on Graphs with Fast Localized Spectral
  Filtering](https://arxiv.org/abs/1606.09375)<br>
    > Michaël Defferrard et al.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)}  + \b^{(k)},
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
        \tilde \L =  \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I.
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - A list of K Chebyshev polynomials of shape
    `[([batch], n_nodes, n_nodes), ..., ([batch], n_nodes, n_nodes)]`; can be computed with
    `spektral.utils.convolution.chebyshev_filter`.

    **Output**

    - Node features with the same shape of the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `K`: order of the Chebyshev polynomials;
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
        K=1,
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
        self.K = K

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(self.K, input_dim, self.channels),
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
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a = inputs

        T_0 = x
        output = K.dot(T_0, self.kernel[0])

        if self.K > 1:
            T_1 = ops.modal_dot(a, x)
            output += K.dot(T_1, self.kernel[1])

        for k in range(2, self.K):
            T_2 = 2 * ops.modal_dot(a, T_1) - T_0
            output += K.dot(T_2, self.kernel[k])
            T_0, T_1 = T_1, T_2

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K}

    @staticmethod
    def preprocess(a):
        a = normalized_laplacian(a)
        a = rescale_laplacian(a)
        return a
