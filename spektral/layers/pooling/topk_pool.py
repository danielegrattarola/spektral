import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.pooling.pool import Pool


class TopKPool(Pool):
    r"""
    A gPool/Top-K layer from the papers

    > [Graph U-Nets](https://arxiv.org/abs/1905.05178)<br>
    > Hongyang Gao and Shuiwang Ji

    and

    > [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)<br>
    > Cătălina Cangea et al.

    **Mode**: single, disjoint.

    This layer computes the following operations:
    $$
        \y = \frac{\X\p}{\|\p\|}; \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$

    where \( \textrm{rank}(\y, K) \) returns the indices of the top K values of
    \(\y\), and \(\p\) is a learnable parameter vector of size \(F\). \(K\) is
    defined for each graph as a fraction of the number of nodes.
    Note that the the gating operation \(\textrm{tanh}(\y)\) (Cangea et al.)
    can be replaced with a sigmoid (Gao & Ji).

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * n_nodes, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes, ratio * n_nodes)`;
    - Reduced graph IDs of shape `(ratio * n_nodes, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * n_nodes, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_mask=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.ratio = ratio
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(
            shape=(self.F, 1),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = "disjoint"
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1])
            self.data_mode = "single"
        if K.ndim(I) == 2:
            I = I[:, 0]
        I = tf.cast(I, tf.int32)

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = self.compute_scores(X, A, I)
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio)
        indices = tf.sort(indices)  # required for ordered SparseTensors
        mask = ops.indices_to_mask(indices, N)

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = (
            0 if len(K.int_shape(A)) == 2 else 1
        )  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.gather(features, indices, axis=axis)

        # Reduce A
        if A_is_sparse:
            A_pooled, _ = ops.gather_sparse_square(A, indices, mask=mask)
        else:
            A_pooled = tf.gather(A, indices, axis=axis)
            A_pooled = tf.gather(A_pooled, indices, axis=axis + 1)

        output = [X_pooled, A_pooled]

        # Reduce I
        if self.data_mode == "disjoint":
            I_pooled = tf.gather(I, indices)
            output.append(I_pooled)

        if self.return_mask:
            output.append(mask)

        return output

    def compute_scores(self, X, A, I):
        return K.dot(X, K.l2_normalize(self.kernel))

    @property
    def config(self):
        return {
            "ratio": self.ratio,
            "return_mask": self.return_mask,
            "sigmoid_gating": self.sigmoid_gating,
        }
