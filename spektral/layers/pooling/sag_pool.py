import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.pooling.topk_pool import TopKPool


class SAGPool(TopKPool):
    r"""
    A self-attention graph pooling layer from the paper

    > [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)<br>
    > Junhyun Lee et al.

    **Mode**: single, disjoint.

    This layer computes:
    $$
        \y = \textrm{GNN}(\A, \X); \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$
    where \(\textrm{rank}(\y, K)\) returns the indices of the top K values of \(\y\) and
    $$
        \textrm{GNN}(\A, \X) = \A \X \W.
    $$

    \(K\) is defined for each graph as a fraction of the number of nodes, controlled by
    the `ratio` argument.

    The gating operation \(\textrm{tanh}(\y)\) (Cangea et al.) can be replaced with a
    sigmoid (Gao & Ji).

    **Input**

    - Node features of shape `(n_nodes_in, n_node_features)`;
    - Adjacency matrix of shape `(n_nodes_in, n_nodes_in)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * n_nodes_in, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes_in, ratio * n_nodes_in)`;
    - Reduced graph IDs of shape `(ratio * n_nodes_in, )` (only in disjoint mode);
    - If `return_selection=True`, the selection mask of shape `(ratio * n_nodes_in, )`.
    - If `return_score=True`, the scoring vector of shape `(n_nodes_in, )`

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_selection`: boolean, whether to return the selection mask;
    - `return_score`: boolean, whether to return the node scoring vector;
    - `sigmoid_gating`: boolean, use a sigmoid activation for gating instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_selection=False,
        return_score=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs,
    ):
        super().__init__(
            ratio,
            return_selection=return_selection,
            return_score=return_score,
            sigmoid_gating=sigmoid_gating,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs,
        )

    def call(self, inputs):
        x, a, i = self.get_inputs(inputs)

        # Graph filter for GNN
        if K.is_sparse(a):
            i_n = tf.sparse.eye(self.n_nodes, dtype=a.dtype)
            a_ = tf.sparse.add(a, i_n)
        else:
            i_n = tf.eye(self.n_nodes, dtype=a.dtype)
            a_ = a + i_n
        fltr = ops.normalize_A(a_)

        y = ops.modal_dot(fltr, K.dot(x, self.kernel))
        output = self.pool(x, a, i, y=y)
        if self.return_score:
            output.append(y)

        return output
