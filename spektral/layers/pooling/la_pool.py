import tensorflow as tf
from scipy import sparse
from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool


class LaPool(SRCPool):
    r"""
    A Laplacian pooling (LaPool) layer from the paper

    > [Towards Interpretable Sparse Graph Representation Learning with Laplacian Pooling](https://arxiv.org/abs/1905.11577)<br>
    > Emmanuel Noutahi et al.

    **Mode**: disjoint.

    This layer computes a soft clustering of the graph by first identifying a set of
    leaders, and then assigning every remaining node to the cluster of the closest
    leader:
    $$
        \V = \|\L\X\|_d; \\
        \i = \{ i \mid \V_i > \V_j, \forall j \in \mathcal{N}(i) \} \\
        \S^\top = \textrm{SparseMax}\left( \beta \frac{\X\X_{\i}^\top}{\|\X\|\|\X_{\i}\|} \right)
    $$
    \(\beta\) is a regularization vecotr that is applied element-wise to the selection
    matrix.
    If `shortest_path_reg=True`, it is equal to the inverse of the shortest path between
    each node and its corresponding leader (this can be expensive since it runs on CPU).
    Otherwise it is equal to 1.

    The reduction and connection are computed as \(\X' = \S\X\) and
    \(\A' = \S^\top\A\S\), respectively.

    Note that the number of nodes in the output graph depends on the input node features.

    **Input**

    - Node features of shape `(n_nodes_in, n_node_features)`;
    - Adjacency matrix of shape `(n_nodes_in, n_nodes_in)`;

    **Output**

    - Reduced node features of shape `(n_nodes_out, channels)`;
    - Reduced adjacency matrix of shape `(n_nodes_out, n_nodes_out)`;
    - If `return_selection=True`, the selection matrix of shape
    `(n_nodes_in, n_nodes_out)`.

    **Arguments**

    - `shortest_path_reg`: boolean, apply the shortest path regularization described in
    the papaer (can be expensive);
    - `return_selection`: boolean, whether to return the selection matrix;
    """

    def __init__(self, shortest_path_reg=True, return_selection=False, **kwargs):
        super().__init__(return_selection=return_selection, **kwargs)

        self.shortest_path_reg = shortest_path_reg

    def call(self, inputs, **kwargs):
        x, a, i = self.get_inputs(inputs)

        # Select leaders
        lap = laplacian(a)
        v = ops.modal_dot(lap, x)
        v = tf.norm(v, axis=-1, keepdims=1)

        row = a.indices[:, 0]
        col = a.indices[:, 1]
        leader_check = tf.cast(tf.gather(v, row) >= tf.gather(v, col), tf.int32)
        leader_mask = ops.scatter_prod(leader_check[:, 0], row, self.n_nodes)
        leader_mask = tf.cast(leader_mask, tf.bool)

        return self.pool(x, a, i, leader_mask=leader_mask)

    def select(self, x, a, i, leader_mask=None):
        # Cosine similarity
        if i is None:
            i = tf.zeros(self.n_nodes, dtype=tf.int32)
        cosine_similarity = sparse_cosine_similarity(x, self.n_nodes, leader_mask, i)

        # Shortest path regularization
        if self.shortest_path_reg:

            def shortest_path(a_):
                return sparse.csgraph.shortest_path(a_, directed=False)

            np_fn_input = tf.sparse.to_dense(a) if K.is_sparse(a) else a
            beta = 1 / tf.numpy_function(shortest_path, [np_fn_input], tf.float64)
            beta = tf.where(tf.math.is_inf(beta), tf.zeros_like(beta), beta)
            beta = tf.boolean_mask(beta, leader_mask, axis=1)
            beta = tf.cast(
                tf.ensure_shape(beta, cosine_similarity.shape), cosine_similarity.dtype
            )
        else:
            beta = 1.0

        s = tf.sparse.softmax(cosine_similarity)
        s = beta * tf.sparse.to_dense(s)

        # Leaders end up entirely in their own cluster
        kronecker_delta = tf.boolean_mask(
            tf.eye(self.n_nodes, dtype=s.dtype), leader_mask, axis=1
        )

        # Create clustering
        s = tf.where(leader_mask[:, None], kronecker_delta, s)

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        return ops.matmul_at_b_a(s, a)

    def reduce_index(self, i, s, leader_mask=None):
        i_pool = tf.boolean_mask(i, leader_mask)

        return i_pool

    def get_config(self):
        config = {"shortest_path_reg": self.shortest_path_reg}
        base_config = super().get_config()
        return {**base_config, **config}


def laplacian(a):
    d = ops.degree_matrix(a, return_sparse_batch=True)
    if K.is_sparse(a):
        a = a.__mul__(-1)
    else:
        a = -a

    return tf.sparse.add(d, a)


def reduce_sum(x, **kwargs):
    if K.is_sparse(x):
        return tf.sparse.reduce_sum(x, **kwargs)
    else:
        return tf.reduce_sum(x, **kwargs)


def sparse_cosine_similarity(x, n_nodes, mask, i):
    mask = tf.cast(mask, tf.int32)
    leader_idx = tf.where(mask)

    # Number of nodes in each graph
    ns = tf.math.segment_sum(tf.ones_like(i), i)
    ks = tf.math.segment_sum(mask, i)

    # s will be a block-diagonal matrix where entry i,j is the cosine
    # similarity between node i and leader j.
    # The code below creates the indices of the sparse block-diagonal matrix
    # Row indices of the block-diagonal S
    starts = tf.cumsum(ns) - ns
    starts = tf.repeat(starts, ks)
    stops = tf.cumsum(ns)
    stops = tf.repeat(stops, ks)
    index_n = tf.ragged.range(starts, stops).flat_values

    # Column indices of the block-diagonal S
    index_k = tf.repeat(leader_idx, tf.repeat(ns, ks))
    index_k_for_s = tf.repeat(tf.range(tf.reduce_sum(ks)), tf.repeat(ns, ks))

    # Make index int64
    index_n = tf.cast(index_n, tf.int64)
    index_k = tf.cast(index_k, tf.int64)
    index_k_for_s = tf.cast(index_k_for_s, tf.int64)

    # Compute similarity between nodes and leaders
    x_n = tf.gather(x, index_n)
    x_n_norm = tf.norm(x_n, axis=-1)
    x_k = tf.gather(x, index_k)
    x_k_norm = tf.norm(x_k, axis=-1)
    values = tf.reduce_sum(x_n * x_k, -1) / (x_n_norm * x_k_norm)

    # Create a sparse tensor for S
    indices = tf.stack((index_n, index_k_for_s), 1)
    s = tf.SparseTensor(
        values=values, indices=indices, dense_shape=(n_nodes, tf.reduce_sum(ks))
    )
    s = tf.sparse.reorder(s)

    return s
