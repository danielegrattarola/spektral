import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv


class GTVConv(Conv):
    r"""
    A graph total variation convolutional layer (GTVConv) from the paper

    > [Total Variation Graph Neural Networks](https://arxiv.org/abs/2211.06218)<br>
    > Jonas Berg Hansen and Filippo Maria Bianchi

    **Mode**: single, disjoint, batch.

    This layer computes
    $$
        \X' = \sigma\left[\left(\I - \delta\L_\hat{\mathbf{\Gamma}}\right) \tilde{\X} \right]
    $$
    where
    $$
        \begin{align}
            \tilde{\X} &= \X \W\\[5pt]
            \L_\hat{\mathbf{\Gamma}} &= \D_\mathbf{\hat{\Gamma}} - \hat{\mathbf{\Gamma}}\\[5pt]
            [\hat{\mathbf{\Gamma}}]_{ij} &= \frac{[\mathbf{A}]_{ij}}{\max\{||\tilde{\x}_i-\tilde{\x}_j||_1, \epsilon\}}\\
        \end{align}
    $$

    **Input**

    - Node features of shape `(batch, n_nodes, n_node_features)`;
    - Adjacency matrix of shape `(batch, n_nodes, n_nodes)`;

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `delta_coeff`: step size for gradient descent of GTV
    - `epsilon`: small number used to numerically stabilize the computation of new adjacency weights
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
        delta_coeff=1.0,
        epsilon=0.001,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
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
            **kwargs,
        )

        self.channels = channels
        self.delta_coeff = delta_coeff
        self.epsilon = epsilon

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
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

    def call(self, inputs, mask=None):
        x, a = inputs

        mode = ops.autodetect_mode(x, a)

        x = K.dot(x, self.kernel)

        if mode == ops.modes.SINGLE:
            output = self._call_single(x, a)

        elif mode == ops.modes.BATCH:
            output = self._call_batch(x, a)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if mask is not None:
            output *= mask[0]

        output = self.activation(output)

        return output

    def _call_single(self, x, a):
        if K.is_sparse(a):
            index_i = a.indices[:, 0]
            index_j = a.indices[:, 1]

            n_nodes = tf.shape(a, out_type=index_i.dtype)[0]

            # Compute absolute differences between neighbouring nodes
            abs_diff = tf.math.abs(
                tf.transpose(tf.gather(x, index_i))
                - tf.transpose(tf.gather(x, index_j))
            )
            abs_diff = tf.math.reduce_sum(abs_diff, axis=0)

            # Compute new adjacency matrix
            gamma = tf.sparse.map_values(
                tf.multiply, a, 1 / tf.math.maximum(abs_diff, self.epsilon)
            )

            # Compute degree matrix from gamma matrix
            d_gamma = tf.sparse.SparseTensor(
                tf.stack([tf.range(n_nodes)] * 2, axis=1),
                tf.sparse.reduce_sum(gamma, axis=-1),
                [n_nodes, n_nodes],
            )

            # Compute laplcian: L = D_gamma - Gamma
            l = tf.sparse.add(d_gamma, tf.sparse.map_values(tf.multiply, gamma, -1.0))

            # Compute adjusted laplacian: L_adjusted = I - delta*L
            l = tf.sparse.add(
                tf.sparse.eye(n_nodes, dtype=x.dtype),
                tf.sparse.map_values(tf.multiply, l, -self.delta_coeff),
            )

            # Aggregate features with adjusted laplacian
            output = ops.modal_dot(l, x)

        else:
            n_nodes = tf.shape(a)[-1]

            abs_diff = tf.math.abs(x[:, tf.newaxis, :] - x)
            abs_diff = tf.reduce_sum(abs_diff, axis=-1)

            gamma = a / tf.math.maximum(abs_diff, self.epsilon)

            degrees = tf.math.reduce_sum(gamma, axis=-1)
            l = -gamma
            l = tf.linalg.set_diag(l, degrees - tf.linalg.diag_part(gamma))
            l = tf.eye(n_nodes, dtype=x.dtype) - self.delta_coeff * l

            output = tf.matmul(l, x)

        return output

    def _call_batch(self, x, a):
        n_nodes = tf.shape(a)[-1]

        abs_diff = tf.reduce_sum(
            tf.math.abs(tf.expand_dims(x, 2) - tf.expand_dims(x, 1)), axis=-1
        )

        gamma = a / tf.math.maximum(abs_diff, self.epsilon)

        degrees = tf.math.reduce_sum(gamma, axis=-1)
        l = -gamma
        l = tf.linalg.set_diag(l, degrees - tf.linalg.diag_part(gamma))
        l = tf.eye(n_nodes, dtype=x.dtype) - self.delta_coeff * l

        output = tf.matmul(l, x)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "delta_coeff": self.delta_coeff,
            "epsilon": self.epsilon,
        }
