import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class MinCutPool(SRCPool):
    r"""
    A MinCut pooling layer from the paper

    > [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481)<br>
    > Filippo Maria Bianchi et al.

    **Mode**: single, batch.

    This layer learns a soft clustering of the input graph as follows:
    $$
    \begin{align}
        \S &= \textrm{MLP}(\X); \\
        \X' &= \S^\top \X \\
        \A' &= \S^\top \A \S; \\
    \end{align}
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron with softmax output.

    Two auxiliary loss terms are also added to the model: the minimum cut loss
    $$
        L_c = - \frac{ \mathrm{Tr}(\S^\top \A \S) }{ \mathrm{Tr}(\S^\top \D \S) }
    $$
    and the orthogonality loss
    $$
        L_o = \left\|
            \frac{\S^\top \S}{\| \S^\top \S \|_F}
            - \frac{\I_K}{\sqrt{K}}
        \right\|_F.
    $$

    The layer can be used without a supervised loss to compute node clustering by
    minimizing the two auxiliary losses.

    **Input**

    - Node features of shape `(batch, n_nodes_in, n_node_features)`;
    - Symmetrically normalized adjacency matrix of shape
    `(batch, n_nodes_in, n_nodes_in)`;

    **Output**

    - Reduced node features of shape `(batch, n_nodes_out, n_node_features)`;
    - Reduced adjacency matrix of shape `(batch, n_nodes_out, n_nodes_out)`;
    - If `return_selection=True`, the selection matrix of shape
    `(batch, n_nodes_in, n_nodes_out)`.

    **Arguments**

    - `k`: number of output nodes;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden layer in
    the MLP used to compute cluster assignments (if `None`, the MLP has only one output
    layer);
    - `mlp_activation`: activation for the MLP layers;
    - `return_selection`: boolean, whether to return the selection matrix;
    - `use_bias`: use bias in the MLP;
    - `kernel_initializer`: initializer for the weights of the MLP;
    - `bias_initializer`: initializer for the bias of the MLP;
    - `kernel_regularizer`: regularization applied to the weights of the MLP;
    - `bias_regularizer`: regularization applied to the bias of the MLP;
    - `kernel_constraint`: constraint applied to the weights of the MLP;
    - `bias_constraint`: constraint applied to the bias of the MLP;
    """

    def __init__(
        self,
        k,
        mlp_hidden=None,
        mlp_activation="relu",
        return_selection=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            return_selection=return_selection,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else []
        self.mlp_activation = mlp_activation

    def build(self, input_shape):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [Dense(self.k, "softmax", **layer_kwargs)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)
        return self.pool(x, a, i, mask=mask)

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        # Orthogonality loss
        ortho_loss = self.orthogonality_loss(s)
        if K.ndim(a) == 3:
            ortho_loss = K.mean(ortho_loss)
        self.add_loss(ortho_loss)

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)

        # MinCut loss
        cut_loss = self.mincut_loss(a, s, a_pool)
        if K.ndim(a) == 3:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Post-processing of A
        a_pool = tf.linalg.set_diag(
            a_pool, tf.zeros(K.shape(a_pool)[:-1], dtype=a_pool.dtype)
        )
        a_pool = ops.normalize_A(a_pool)

        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    def orthogonality_loss(self, s):
        ss = ops.modal_dot(s, s, transpose_a=True)
        i_s = tf.eye(self.k, dtype=ss.dtype)
        ortho_loss = tf.norm(
            ss / tf.norm(ss, axis=(-1, -2), keepdims=True) - i_s / tf.norm(i_s),
            axis=(-1, -2),
        )
        return ortho_loss

    @staticmethod
    def mincut_loss(a, s, a_pool):
        num = tf.linalg.trace(a_pool)
        d = ops.degree_matrix(a)
        den = tf.linalg.trace(ops.matmul_at_b_a(s, d))
        cut_loss = -(num / den)
        return cut_loss

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }
        base_config = super().get_config()
        return {**base_config, **config}
