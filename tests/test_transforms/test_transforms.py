import numpy as np
import scipy.sparse as sp

from spektral.data import Graph
from spektral import transforms as tr

N = 10
F = 3
S = 4
n_labels = 2
x = np.ones((N, F))
a = sp.csr_matrix(np.ones((N, N)))
e = np.ones((N * N, S))
y_gl = np.ones(n_labels)
y_nl = np.ones((N, n_labels))
y_sc = 1


g_gl = Graph(x=x, a=a, e=e, y=y_gl)
g_nl = Graph(x=x, a=a, e=e, y=y_nl)
g_sc = Graph(x=x, a=a, e=e, y=y_sc)


def test_adj_to_sp_tensor():
    t = tr.AdjToSpTensor()
    g = Graph(x=x, a=a, e=e, y=y_gl)
    assert callable(t)
    t(g)


def test_clustering_coeff():
    t = tr.ClusteringCoeff()
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)
    g = Graph(x=None, a=a, e=e, y=y_gl)
    t(g)


def test_constant():
    t = tr.Constant(10)
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)
    g = Graph(x=None, a=a, e=e, y=y_gl)
    t(g)


def test_degree():
    t = tr.Degree(10)
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)
    g = Graph(x=None, a=a, e=e, y=y_gl)
    t(g)


def test_delaunay():
    t = tr.Delaunay()
    assert callable(t)
    x = np.random.rand(N, 2)
    g = Graph(x=x, a=a, e=e, y=y_nl)
    t(g)
    g = Graph(x=x, a=a.A, e=e, y=y_nl)
    t(g)


def test_gcn_filter():
    t = tr.GCNFilter()
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_nl)
    t(g)
    g = Graph(x=x, a=a.A, e=e, y=y_nl)
    t(g)


def test_layer_preprocess():
    from spektral.layers import GCNConv
    t = tr.LayerPreprocess(GCNConv)
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_nl)
    t(g)


def test_normalize_adj():
    t = tr.NormalizeAdj()
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_nl)
    t(g)
    g = Graph(x=x, a=a.A, e=e, y=y_nl)
    t(g)


def test_normalize_one():
    t = tr.NormalizeOne()
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)


def test_normalize_sphere():
    t = tr.NormalizeSphere()
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)


def test_one_hot():
    t = tr.OneHotLabels(depth=2)
    assert callable(t)
    g = Graph(x=x, a=a, e=e, y=y_gl)
    t(g)
    g = Graph(x=x, a=a, e=e, y=y_nl)
    t(g)
    g = Graph(x=x, a=a, e=e, y=y_sc)
    t(g)
