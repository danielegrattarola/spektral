import numpy as np

from spektral.data.graph import Graph

N = 5
F = 4
S = 3
n_out = 2


def _check_graph(x, a, e, y):
    g = Graph()
    g = Graph(x=x)
    g = Graph(a=a)
    g = Graph(x=x, a=a, e=e, y=y)

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


def test_graph():
    x = np.ones((N, F))
    a = np.ones((N, N))
    e = np.ones((N, N, S))
    y = np.ones((n_out,))

    _check_graph(x, a, e, y)
