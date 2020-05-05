import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K
from tensorflow.keras.layers import Layer, Dense

from spektral.layers import ops


class MinCutPool(Layer):
    r"""
    A minCUT pooling layer as presented by
    [Bianchi et al. (2019)](https://arxiv.org/abs/1907.00481).

    **Mode**: batch.

    This layer computes a soft clustering \(\S\) of the input graphs using a MLP,
    and reduces graphs as follows:

    $$
        \S = \textrm{MLP}(\X); \\
        \A' = \S^\top \A \S; \X' = \S^\top \X;
    $$

    where MLP is a multi-layer perceptron with softmax output.
    Two auxiliary loss terms are also added to the model: the _minCUT loss_
    $$
        - \frac{ \mathrm{Tr}(\S^\top \A \S) }{ \mathrm{Tr}(\S^\top \D \S) }
    $$
    and the _orthogonality loss_
    $$
        \left\|
            \frac{\S^\top \S}{\| \S^\top \S \|_F}
            - \frac{\I_K}{\sqrt{K}}
        \right\|_F.
    $$

    The layer can be used without a supervised loss, to compute node clustering
    simply by minimizing the two auxiliary losses.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Binary adjacency matrix of shape `([batch], N, N)`;

    **Output**

    - Reduced node features of shape `([batch], K, F)`;
    - Reduced adjacency matrix of shape `([batch], K, K)`;
    - If `return_mask=True`, the soft clustering matrix of shape `([batch], N, K)`.

    **Arguments**

    - `k`: number of nodes to keep;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP used to compute cluster assignments (if None, the MLP has
    only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `return_mask`: boolean, whether to return the cluster assignment matrix;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(self,
                 k,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 return_mask=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        mlp_layers = []
        for i, channels in enumerate(self.mlp_hidden):
            mlp_layers.append(
                Dense(channels, self.mlp_activation, **layer_kwargs)
            )
        mlp_layers.append(
            Dense(self.k, 'softmax', **layer_kwargs)
        )
        self.mlp = Sequential(mlp_layers)

        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X, A = inputs
            I = None

        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(X) == 3

        # Compute cluster assignment matrix
        S = self.mlp(X)

        # MinCut regularization
        A_pooled = ops.matmul_AT_B_A(S, A)
        num = tf.linalg.trace(A_pooled)
        D = ops.degree_matrix(A)
        den = tf.linalg.trace(ops.matmul_AT_B_A(S, D)) + K.epsilon()
        cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Orthogonality regularization
        SS = ops.matmul_AT_B(S, S)
        I_S = tf.eye(self.k, dtype=SS.dtype)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2), keepdims=True) - I_S / tf.norm(I_S),
            axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(ortho_loss)
        self.add_loss(ortho_loss)

        # Pooling
        X_pooled = ops.matmul_AT_B(S, X)
        A_pooled = tf.linalg.set_diag(
            A_pooled, tf.zeros(K.shape(A_pooled)[:-1], dtype=A_pooled.dtype)
        )  # Remove diagonal
        A_pooled = ops.normalize_A(A_pooled)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.math.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        X_shape_out = X_shape[:-2] + (self.k,) + X_shape[-1:]
        A_shape_out = A_shape[:-2] + (self.k, self.k)

        output_shape = [X_shape_out, A_shape_out]

        if len(input_shape) == 3:
            I_shape_out = A_shape[:-2] + (self.k, )
            output_shape.append(I_shape_out)

        if self.return_mask:
            S_shape_out = A_shape[:-1] + (self.k, )
            output_shape.append(S_shape_out)

        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': self.mlp_activation,
            'return_mask': self.return_mask,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))