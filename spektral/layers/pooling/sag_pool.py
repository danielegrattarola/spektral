from tensorflow.keras import backend as K

from spektral.layers.pooling.topk_pool import ops, TopKPool


class SAGPool(TopKPool):
    r"""
    A self-attention graph pooling layer as presented by
    [Lee et al. (2019)](https://arxiv.org/abs/1904.08082).

    **Mode**: single, disjoint.

    This layer computes the following operations:

    $$
    \y = \textrm{GNN}(\A, \X); \;\;\;\;
    \i = \textrm{rank}(\y, K); \;\;\;\;
    \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
    \A' = \A_{\i, \i}
    $$

    where \( \textrm{rank}(\y, K) \) returns the indices of the top K values of
    \(\y\), and \(\textrm{GNN}\) consists of one GraphConv layer with no
    activation. \(K\) is defined for each graph as a fraction of the number of
    nodes.

    This layer temporarily makes the adjacency matrix dense in order to compute
    \(\A'\).
    If memory is not an issue, considerable speedups can be achieved by using
    dense graphs directly.
    Converting a graph from sparse to dense and back to sparse is an expensive
    operation.

    **Input**

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * N, F)`;
    - Reduced adjacency matrix of shape `(ratio * N, ratio * N)`;
    - Reduced graph IDs of shape `(ratio * N, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * N, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(self,
                 ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super().__init__(ratio,
                         return_mask=return_mask,
                         sigmoid_gating=sigmoid_gating,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         kernel_constraint=kernel_constraint,
                         **kwargs)

    def compute_scores(self, X, A, I):
        scores = K.dot(X, self.kernel)
        scores = ops.filter_dot(A, scores)
        return scores