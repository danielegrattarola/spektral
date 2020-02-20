import tensorflow as tf
from tensorflow.keras import backend as K, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

from spektral.layers import ops


class SAGPool(Layer):
    r"""
    A self-attention graph pooling layer as presented by
    [Lee et al. (2019)](https://arxiv.org/abs/1904.08082) and
    [Knyazev et al. (2019)](https://arxiv.org/abs/1905.02850).

    This layer computes the following operations:

    $$
    \y = \textrm{GNN}(\A, \X); \;\;\;\;
    \i = \textrm{rank}(\y, K); \;\;\;\;
    \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
    \A' = \A^2_{\i, \i}
    $$

    where \( \textrm{rank}(\y, K) \) returns the indices of the top K values of
    \(\y\), and \(\textrm{GNN}\) consists of one GraphConv layer with no
    activation.

    Due to the lack of sparse-sparse matrix multiplication support, this layer
    temporarily makes the adjacency matrix dense in order to compute \(\A^2\)
    (needed to preserve connectivity after pooling).

    **If memory is not an issue, considerable speedups can be achieved by using
    dense graphs directly.
    Converting a graph from sparse to dense and back to sparse is an expensive
    operation.**

    **Mode**: single, disjoint.

    **Input**

    - node features of shape `(n_nodes, n_features)`;
    - adjacency matrix of shape `(n_nodes, n_nodes)`;
    - (optional) graph IDs of shape `(n_nodes, )` (disjoint mode);

    **Output**

    - reduced node features of shape `(n_graphs * k, n_features)`;
    - reduced adjacency matrix of shape `(n_graphs * k, n_graphs * k)`;
    - reduced graph IDs with shape `(n_graphs * k, )` (disjoint mode);
    - If `return_mask=True`, the binary mask used for pooling, with shape
    `(n_graphs * k, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    """

    def __init__(self, ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(shape=(self.F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.top_k_var = tf.Variable(0.0,
                                     trainable=False,
                                     validate_shape=False,
                                     shape=tf.TensorShape(None))
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = 'graph'
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1])
            self.data_mode = 'single'
        if K.ndim(I) == 2:
            I = I[:, 0]
        I = tf.cast(I, tf.int32)

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, self.kernel)
        y = ops.filter_dot(A, y)
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio, self.top_k_var)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices), (N,))

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = 0 if len(K.int_shape(A)) == 2 else 1  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.boolean_mask(features, mask, axis=axis)

        # Compute A^2
        if A_is_sparse:
            A_dense = tf.sparse.to_dense(A)
        else:
            A_dense = A
        A_squared = K.dot(A, A_dense)

        # Reduce A
        A_pooled = tf.boolean_mask(A_squared, mask, axis=axis)
        A_pooled = tf.boolean_mask(A_pooled, mask, axis=axis + 1)
        if A_is_sparse:
            A_pooled = ops.dense_to_sparse(A_pooled)

        output = [X_pooled, A_pooled]

        # Reduce I
        if self.data_mode == 'graph':
            I_pooled = tf.boolean_mask(I[:, None], mask)[:, 0]
            output.append(I_pooled)

        if self.return_mask:
            output.append(mask)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_mask:
            output_shape += [(input_shape[0][:-1])]
        return output_shape

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'return_mask': self.return_mask,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))