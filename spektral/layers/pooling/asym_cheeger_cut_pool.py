import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class AsymCheegerCutPool(SRCPool):
    r"""
    An Asymmetric Cheeger Cut Pooling layer from the paper
    > [Total Variation Graph Neural Networks](https://arxiv.org/abs/2211.06218)<br>
    > Jonas Berg Hansen and Filippo Maria Bianchi

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

    The layer includes two auxiliary loss terms/components:
    A graph total variation component given by
    $$
        L_\text{GTV} = \frac{1}{2E} \sum_{k=1}^K \sum_{i=1}^N \sum_{j=i}^N a_{i,j} |s_{i,k} - s_{j,k}|,
    $$
    where \(E\) is the number of edges/links, \(K\) is the number of clusters or output nodes, and \(N\) is the number of nodes.
    
    An asymmetrical norm component given by
    $$
        L_\text{AN} = \frac{N(K - 1) - \sum_{k=1}^K ||\s_{:,k} - \textrm{quant}_{K-1} (\s_{:,k})||_{1, K-1}}{N(K-1)},
    $$

    The layer can be used without a supervised loss to compute node clustering by
    minimizing the two auxiliary losses.

    **Input**

    - Node features of shape `(batch, n_nodes_in, n_node_features)`;
    - Adjacency matrix of shape `(batch, n_nodes_in, n_nodes_in)`;

    **Output**

    - Reduced node features of shape `(batch, n_nodes_out, n_node_features)`;
    - If `return_selection=True`, the selection matrix of shape 
    `(batch, n_nodes_in, n_nodes_out)`.

    **Arguments**

    - `k`: number of output nodes;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden layer in
    the MLP used to compute cluster assignments (if `None`, the MLP has only one output
    layer);
    - `mlp_activation`: activation for the MLP layers;
    - `totvar_coeff`: coefficient for graph total variation loss component;
    - `balance_coeff`: coefficient for asymmetric norm loss component;  
    - `return_selection`: boolean, whether to return the selection matrix;
    - `use_bias`: use bias in the MLP;
    - `kernel_initializer`: initializer for the weights of the MLP;
    - `bias_regularizer`: regularization applied to the bias of the MLP;
    - `kernel_constraint`: constraint applied to the weights of the MLP;
    - `bias_constraint`: constraint applied to the bias of the MLP;
    """

    def __init__(
        self,
        k,
        mlp_hidden=None,
        mlp_activation="relu",
        totvar_coeff=1.0,
        balance_coeff=1.0,
        return_selection=False,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            k=k,
            mlp_hidden=mlp_hidden,
            mlp_activation=mlp_activation,
            return_selection=return_selection,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

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

        # Total variation loss
        tv_loss = self.totvar_loss(a, s)
        if K.ndim(a) == 3:
            tv_loss = K.mean(tv_loss)
        self.add_loss(self.totvar_coeff * tv_loss)

        # Asymmetric l1-norm loss
        bal_loss = self.balance_loss(s)
        if K.ndim(a) == 3:
            bal_loss = K.mean(bal_loss)
        self.add_loss(self.balance_coeff * bal_loss)

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)

        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool

    def totvar_loss(self, a, s):
        if K.is_sparse(a):
            index_i = a.indices[:, 0]
            index_j = a.indices[:, 1]

            n_edges = tf.cast(len(a.values), dtype=s.dtype)

            loss = tf.math.reduce_sum(
                a.values[:, tf.newaxis]
                * tf.math.abs(tf.gather(s, index_i) - tf.gather(s, index_j)),
                axis=(-2, -1),
            )

        else:
            n_edges = tf.cast(tf.math.count_nonzero(a, axis=(-2, -1)), dtype=s.dtype)
            n_nodes = tf.shape(a)[-1]
            if K.ndim(a) == 3:
                loss = tf.math.reduce_sum(
                    a
                    * tf.math.reduce_sum(
                        tf.math.abs(
                            s[:, tf.newaxis, ...]
                            - tf.repeat(s[..., tf.newaxis, :], n_nodes, axis=-2)
                        ),
                        axis=-1,
                    ),
                    axis=(-2, -1),
                )
            else:
                loss = tf.math.reduce_sum(
                    a
                    * tf.math.reduce_sum(
                        tf.math.abs(
                            s - tf.repeat(s[..., tf.newaxis, :], n_nodes, axis=-2)
                        ),
                        axis=-1,
                    ),
                    axis=(-2, -1),
                )

        loss *= 1 / (2 * n_edges)

        return loss

    def balance_loss(self, s):
        n_nodes = tf.cast(tf.shape(s, out_type=tf.int32)[-2], s.dtype)

        # k-quantile
        idx = tf.cast(tf.math.floor(n_nodes / self.k) + 1, dtype=tf.int32)
        med = tf.math.top_k(tf.linalg.matrix_transpose(s), k=idx).values[..., -1]
        # Asymmetric l1-norm
        if K.ndim(s) == 2:
            loss = s - med
        else:
            loss = s - med[:, tf.newaxis, ...]
        loss = (tf.cast(loss >= 0, loss.dtype) * (self.k - 1) * loss) + (
            tf.cast(loss < 0, loss.dtype) * loss * -1.0
        )
        loss = tf.math.reduce_sum(loss, axis=(-2, -1))
        loss = 1 / (n_nodes * (self.k - 1)) * (n_nodes * (self.k - 1) - loss)

        return loss

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
            "totvar_coeff": self.totvar_coeff,
            "balance_coeff": self.balance_coeff,
        }
        base_config = super().get_config()
        return {**base_config, **config}
