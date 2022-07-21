import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class DMoNPool(SRCPool):
    r"""
    The DMoN pooling layer from the paper

    > [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904)<br>
    > Anton Tsitsulin et al.

    **Mode**: single, batch.

    This layer learns a soft clustering of the input graph as follows:
    $$
    \begin{align}
        \C &= \textrm{MLP}(\X); \\
        \X' &= \C^\top \X \\
        \A' &= \C^\top \A \C; \\
    \end{align}
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron with softmax output.

    Two auxiliary loss terms are also added to the model: the modularity loss
    $$
        L_m = - \frac{1}{2m} \mathrm{Tr}(\C^\top \A \C - \C^\top \d^\top \d \C)
    $$
    and the collapse regularization loss
    $$
        L_c = \frac{\sqrt{k}}{n} \left\|
            \sum_i \C_i^\top
        \right\|_F -1.
    $$

    This layer is based on the original implementation found
    [here](https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py).

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
    - `collapse_regularization`: strength of the collapse regularization;
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
        collapse_regularization=0.1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            k=k,
            mlp_hidden=mlp_hidden,
            mlp_activation=mlp_activation,
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
        self.collapse_regularization = collapse_regularization

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

        # Collapse loss
        col_loss = self.collapse_loss(a, s)
        if K.ndim(a) == 3:
            col_loss = K.mean(col_loss)
        self.add_loss(self.collapse_regularization * col_loss)

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)

        # Modularity loss
        mod_loss = self.modularity_loss(a, s, a_pool)
        if K.ndim(a) == 3:
            mod_loss = K.mean(mod_loss)
        self.add_loss(mod_loss)

        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    def modularity_loss(self, a, s, a_pool):

        if K.is_sparse(a):
            n_edges = tf.cast(len(a.values), dtype=s.dtype)

            degrees = tf.sparse.reduce_sum(a, axis=-1)
            degrees = tf.reshape(degrees, (-1, 1))
        else:
            n_edges = tf.cast(tf.math.count_nonzero(a, axis=(-2, -1)), dtype=s.dtype)
            degrees = tf.reduce_sum(a, axis=-1, keepdims=True)

        normalizer_left = tf.matmul(s, degrees, transpose_a=True)
        normalizer_right = tf.matmul(degrees, s, transpose_a=True)

        if K.ndim(s) == 3:
            normalizer = (
                ops.modal_dot(normalizer_left, normalizer_right)
                / 2
                / tf.reshape(n_edges, [tf.shape(n_edges)[0]] + [1] * 2)
            )
        else:
            normalizer = ops.modal_dot(normalizer_left, normalizer_right) / 2 / n_edges

        loss = -tf.linalg.trace(a_pool - normalizer) / 2 / n_edges

        return loss

    def collapse_loss(self, a, s):
        cluster_sizes = tf.math.reduce_sum(s, axis=-2)
        n_nodes = tf.cast(tf.shape(a)[-1], s.dtype)
        loss = (
            tf.norm(cluster_sizes, axis=-1)
            / n_nodes
            * tf.sqrt(tf.cast(self.k, s.dtype))
            - 1
        )

        return loss

    def get_config(self):
        config = {
            "collapse_regularization": self.collapse_regularization,
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }
        base_config = super().get_config()
        return {**base_config, **config}
