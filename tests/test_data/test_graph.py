import numpy as np

from spektral.data.graph import Graph

n_nodes = 5
n_node_features = 4
n_edge_features = 3
n_out = 2


def _check_graph(x, a, e, y):
    g = Graph()  # Empty graph
    g = Graph(x=x)  # Only node features
    g = Graph(a=a)  # Only adjacency
    g = Graph(x=x, a=a, e=e, y=y, extra=1)  # Complete graph with extra attribute

    # numpy
    g_np = g.numpy()
    g_gt_names = ["x", "a", "e", "y"]
    g_gt = [x, a, e, y]
    for i in range(len(g_gt)):
        assert np.all(g_np[i] == g_gt[i])

    # __getitem__
    for i in range(len(g_gt)):
        assert np.all(g.__getitem__(g_gt_names[i]) == g_gt[i])

    # __repr__
    print(g)

    # Properties
    assert g.n_nodes == n_nodes
    assert g.n_node_features == n_node_features
    assert g.n_edge_features == n_edge_features
    assert g.n_labels == n_out
    assert g.n_edges == np.count_nonzero(a)


def test_graph():
    x = np.ones((n_nodes, n_node_features))
    a = np.ones((n_nodes, n_nodes))
    e = np.ones((n_nodes, n_nodes, n_edge_features))
    y = np.ones((n_out,))

    _check_graph(x, a, e, y)
