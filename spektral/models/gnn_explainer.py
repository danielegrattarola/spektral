import networkx as nx
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from spektral.layers.ops import dot, sp_matrix_to_sp_tensor


class GNNExplainer:
    """
    The GNNExplainer model from the paper:
    > [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)<br>
    > Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik and Jure Leskovec.
    The model can be used to explain the predictions for a single node or for an entire
    graph. In both cases, it returns the subgraph that mostly contribute to the
    prediction.
    **Arguments**
    - `model`: tf.keras.Model to explain;
    - `num_conv_layers`: number of graph convolutional layers in `model`;
    - `x`: feature matrix of shape `(n_nodes, n_node_features)`;
    - `a`: sparse adjacency matrix `(n_nodes, n_nodes)`;
    - `mode`: either `node` to explain single node prediction or `graph` to explain the
    prediction for the whole graph;
    - `node_idx`: index of the node to explain;
    - `verbose`: `True` to print loss values during the training, else `False`;
    - `epochs`: number of epochs to train the explainer;
    """

    def __init__(
        self,
        model,
        x,
        a,
        num_conv_layers=None,
        adj_transf=None,
        mode="node",
        verbose=False,
        epochs=100,
    ):
        self.model = model
        self.num_conv_layers = num_conv_layers
        self.a = a
        self.adj_transf = adj_transf
        self.x = x
        self.mode = mode
        self.verbose = verbose
        self.epochs = epochs

    def explain_node(
        self,
        node_idx=None,
        edge_size_reg=0.0005,
        feat_size_reg=0.1,
        edge_entropy_reg=0.1,
        feat_entropy_reg=0.1,
        laplacian_reg=0.0,
    ):
        """
        Method used to start the training.
        **Arguments**
        - `edge_size_reg`: controls the edge size of the subgraph that contributes to
        the prediction;
        - `feat_size_reg`: controls the number of features that contribute to the
        prediction;
        - `edge_entropy_reg`: controls the discretization of the adjacency mask;
        - `feat_entropy_reg`: controls the discretization of the feature mask;
        - `laplacian_reg`: controls the laplacian loss;
        """
        self.node_idx = node_idx
        self.edge_size_reg = edge_size_reg
        self.feat_size_reg = feat_size_reg
        self.edge_entropy_reg = edge_entropy_reg
        self.feat_entropy_reg = feat_entropy_reg
        self.laplacian_reg = laplacian_reg

        # get the computational graph
        if self.mode == "node":
            self.comp_graph = k_hop_sparse_subgraph(
                self.a, self.node_idx, self.num_conv_layers, self.adj_transf
            )

        elif self.mode == "graph":
            self.comp_graph = tf.cast(self.a, tf.float32)

        # predictions needed to compute the explainer's loss
        if self.mode == "node":
            self.y_pred = tf.argmax(
                self.model([self.x, self.a], training=False), axis=1
            )

        elif self.mode == "graph":
            self.i = tf.zeros(self.x.shape[0], dtype=tf.int32)
            self.y_pred = tf.argmax(
                self.model([self.x, self.a, self.i], training=False), axis=1
            )

        # init the optimizer for the training
        self.opt = tf.keras.optimizers.Adam(0.01)

        if self.mode == "node":
            self.node_pred = self.y_pred[self.node_idx]

        elif self.mode == "graph":
            self.node_pred = self.y_pred[0]

        self.y_pred = tf.cast(self.y_pred, tf.float32)

        # init the trainable adj mask
        adj_mask = tf.Variable(
            tf.random.normal(
                self.comp_graph.values.shape, stddev=(2 / self.x.shape[0]) ** 0.5
            ),
            dtype=tf.float32,
            trainable=True,
        )

        # init the trainable feature mask
        feat_mask = tf.Variable(
            tf.random.normal((1, self.x.shape[1]), stddev=0.1),
            dtype=tf.float32,
            trainable=True,
        )

        # training loop
        for i in range(self.epochs):
            losses = self._train_step(adj_mask, feat_mask)
            print(
                {key: val.numpy() for key, val in losses.items()}
            ) if self.verbose else None
        return adj_mask, feat_mask

    @tf.function
    def _train_step(self, adj_mask, feat_mask):
        with tf.GradientTape() as tape:
            # compute the masked adj matrix
            masked_adj = tf.sparse.map_values(
                tf.multiply, self.comp_graph, tf.nn.sigmoid(adj_mask)
            )

            # compute the masked feature matrix
            masked_input = self.x * tf.nn.sigmoid(feat_mask)

            if self.mode == "node":
                pred = self.model([masked_input, masked_adj], training=False)[
                    self.node_idx, self.node_pred
                ]

            elif self.mode == "graph":
                pred = self.model([masked_input, masked_adj, self.i], training=False)[
                    0, self.node_pred
                ]

            loss, losses = self._explain_loss_fn(pred, adj_mask, feat_mask)
        grad = tape.gradient(loss, [adj_mask, feat_mask])
        self.opt.apply_gradients(zip(grad, [adj_mask, feat_mask]))
        return losses

    def _explain_loss_fn(self, y_pred, adj_mask, feat_mask):
        mask = tf.nn.sigmoid(adj_mask)

        # prediction and entropy loss
        pred_loss = -tf.math.log(y_pred + 1e-15)
        edge_size_loss = self.edge_size_reg * tf.reduce_sum(mask)
        entropy = -mask * tf.math.log(mask + 1e-15) - (1 - mask) * tf.math.log(
            1 - mask + 1e-15
        )
        edge_entropy_loss = self.edge_entropy_reg * tf.reduce_mean(entropy)

        # smoothness of signal loss
        if self.mode == "node":
            masked_adj = tf.sparse.map_values(tf.multiply, self.comp_graph, mask)
            d = tf.linalg.diag(tf.sparse.reduce_sum(masked_adj, axis=0))
            masked_adj = tf.sparse.map_values(tf.multiply, masked_adj, -1)

            laplacian = tf.sparse.add(d, masked_adj)
            laplacian = tf.cast(laplacian, tf.float32)
            quad_form = (
                tf.reshape(self.y_pred, (1, -1))
                @ laplacian
                @ tf.reshape(self.y_pred, (-1, 1))
            )
            smoothness_loss = self.laplacian_reg * quad_form
        elif self.mode == "graph":
            smoothness_loss = 0

        # feature loss
        mask = tf.nn.sigmoid(feat_mask)
        feat_size_loss = self.feat_size_reg * tf.reduce_sum(mask)
        entropy = -mask * tf.math.log(mask + 1e-15) - (1 - mask) * tf.math.log(
            1 - mask + 1e-15
        )
        feat_entropy_loss = self.feat_entropy_reg * tf.reduce_mean(entropy)

        loss = pred_loss + edge_size_loss + edge_entropy_loss + smoothness_loss
        loss += feat_size_loss + feat_entropy_loss

        losses = {
            "pred_loss": pred_loss,
            "edge_size_loss": edge_size_loss,
            "ent_edge_loss": edge_entropy_loss,
            "smooth_loss": smoothness_loss,
            "feat_size_loss": feat_size_loss,
            "ent_feat_loss": feat_entropy_loss,
        }
        return loss, losses

    def _explainer_cleaning(self, adj_mask, feat_mask, adj_tr, top_feat, k):
        # get the masks
        selected_adj_mask = tf.nn.sigmoid(adj_mask)
        selected_feat_mask = tf.nn.sigmoid(feat_mask)

        # convert into a binary matrix
        if self.adj_transf:
            comp_graph_values = tf.ones_like(self.comp_graph.values)
            self.comp_graph = tf.sparse.SparseTensor(
                self.comp_graph.indices, comp_graph_values, self.comp_graph.shape
            )

        # get the final masked adj matrix
        selected_subgraph = tf.sparse.map_values(
            tf.multiply, self.comp_graph, selected_adj_mask
        )

        # impose the symmetry of the adj matrix
        selected_subgraph = (
            tf.sparse.add(selected_subgraph, tf.sparse.transpose(selected_subgraph)) / 2
        )

        # remove the edges which value is < adj_tr
        selected_adj_mask = tf.where(
            selected_subgraph.values >= adj_tr, selected_subgraph.values, 0
        )

        selected_subgraph = tf.sparse.map_values(
            tf.multiply, self.comp_graph, selected_adj_mask
        )

        if self.mode == "node":
            # get the final denoised subgraph centerd in the interested node
            selected_subgraph = k_hop_sparse_subgraph(
                selected_subgraph, self.node_idx, k
            )

        # the the top_feat relevant feature ids
        selected_features = tf.argsort(
            tf.nn.sigmoid(selected_feat_mask), direction="DESCENDING"
        )[0][:top_feat]
        return selected_subgraph, selected_features

    def plot_subgraph(self, adj_mask, feat_mask, adj_tr=0.1, top_feat=10, k=2):
        """
        Method used to clean the important subgraph and features and make the plots.
        **Arguments**
        - `adj_tr`: threshold needed to remove low importance nodes;
        - `top_feat`: number of features to return sorted by importance;
        - `k`: order of neighbors of the subgraph around `node_idx` to ;
        """
        adj_mtx, top_ftrs = self._explainer_cleaning(
            adj_mask, feat_mask, adj_tr, top_feat, k
        )

        edge_list = adj_mtx.indices.numpy()
        weights = adj_mtx.values

        G = nx.Graph()
        for i, (n1, n2) in enumerate(edge_list):
            if weights[i] != 0:
                G.add_edge(n1, n2, w=weights[i].numpy())

        # take the largest component
        giant = max(nx.algorithms.components.connected_components(G), key=len)

        pos = nx.layout.spring_layout(G, k=0.04)
        nx.draw_networkx_nodes(G, pos=pos, node_size=30, nodelist=giant)
        nx.draw_networkx_edges(G, pos=pos, edge_color="grey", alpha=0.8)
        nx.draw_networkx_labels(
            G, pos=pos, font_color="black", font_size=10, verticalalignment="bottom"
        )

        print("Top features: ", top_ftrs.numpy())

        return G


def k_hop_sparse_subgraph(a, node_idx, k, transformer=None):
    """
    Computes the subgraph containing all the neighbors of `node_idx` up to the k-th order.
    If `a` is not the binary adjacency matrix a  `transformer` should be passed.
    **Arguments**
    - `a`: sparse `(n_nodes, n_nodes)` graph tensor;
    - `node_idx`: center node;
    - `k`: order of neighbor;
    - `transformer`: one of the functions from the `spektral.transforms` module,
       needed to convert the binary adjacency matrix into the correct format for the model;
    """
    if a.dtype != tf.float32:
        a = tf.cast(a, tf.float32)

    if transformer:
        a = binary_adj_converter(a)

    power_a = tf.sparse.eye(a.shape[0])
    k_neighs = np.zeros(a.shape[0]).astype("float32").reshape(1, -1)
    k_neighs[0, node_idx] = 1

    for _ in range(k - 1):
        power_a = dot(power_a, a)
        temp = tf.sparse.slice(power_a, start=[node_idx, 0], size=[1, power_a.shape[0]])
        k_neighs += tf.sparse.to_dense(temp)

    comp_graph = tf.sparse.add(a * tf.reshape(k_neighs, (-1, 1)), a * k_neighs)
    is_nonzero = tf.not_equal(comp_graph.values, 0)
    comp_graph = tf.sparse.retain(comp_graph, is_nonzero)
    comp_graph = tf.sign(comp_graph)

    if transformer:
        comp_graph = sp_tensor_to_sp_matrix(comp_graph)
        comp_graph = transformer(comp_graph)
        return sp_matrix_to_sp_tensor(comp_graph)
    else:
        return comp_graph


def binary_adj_converter(a_in):
    """
    Transforms a graph matrix into the binary adjacency matrix.
    **Arguments**
    - `a_in`: sparse `(n_nodes, n_nodes)` graph tensor;
    """
    a_idx = a_in.indices
    off_diag_idx = tf.not_equal(a_idx[:, 0], a_idx[:, 1])
    a_idx = a_idx[off_diag_idx]

    a = tf.sparse.SparseTensor(
        a_idx, tf.ones(a_idx.shape[0], dtype=tf.float32), a_in.shape
    )
    return a


def sp_tensor_to_sp_matrix(a):
    """
    Transforms a sparse tensor into a sparse scipy matrix .
    **Arguments**
    - `a`: sparse `(n_nodes, n_nodes)` graph tensor;
    """
    a_idx = a.indices
    a_val = a.values

    row_idx = a_idx[:, 0]
    col_idx = a_idx[:, 1]

    return csr_matrix((a_val, (row_idx, col_idx)), shape=a.shape)
