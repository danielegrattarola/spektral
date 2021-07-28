import numpy as np

from spektral import transforms
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph

n_graphs = 10
Ns = np.random.randint(3, 8, n_graphs)
f = 3
s = 3


def test_dataset():
    class TestDataset(Dataset):
        def read(self):
            return [
                Graph(
                    x=np.random.rand(n, f),
                    a=np.random.randint(0, 2, (n, n)),
                    e=np.random.rand(n, n, s),
                    y=np.array([0.0, 1.0]),
                )
                for n in Ns
            ]

    d = TestDataset()

    assert d.n_node_features == f
    assert d.n_edge_features == s
    assert d.n_labels == 2

    # signature
    for k in ["x", "a", "e", "y"]:
        assert k in d.signature

    # __getitem__
    assert isinstance(d[0], Graph)
    assert isinstance(d[:3], Dataset)
    assert isinstance(d[[1, 3, 4]], Dataset)

    # __setitem__
    n = 100
    g = Graph(
        x=np.random.rand(n, f),
        a=np.random.randint(0, 2, (n, n)),
        e=np.random.rand(n, n, s),
        y=np.array([0.0, 1.0]),
    )

    # single assignment
    d[0] = g
    assert d[0].n_nodes == n and all([d_.n_nodes != n for d_ in d[1:]])

    # Slice assignment
    d[1:3] = [g] * 2
    assert (
        d[1].n_nodes == n
        and d[2].n_nodes == n
        and all([d_.n_nodes != n for d_ in d[3:]])
    )

    # List assignment
    d[[3, 4]] = [g] * 2
    assert (
        d[3].n_nodes == n
        and d[4].n_nodes == n
        and all([d_.n_nodes != n for d_ in d[5:]])
    )

    # __len__
    assert d.__len__() == n_graphs

    # __add__
    d2 = TestDataset()
    assert len(d + d2) == len(d) + len(d2)
    assert len(d + d2) == len(d2 + d)  # Test no side effects by repeated calling

    # __repr__
    print(d)

    # Test that shuffling doesn't crash
    np.random.shuffle(d)

    # Test apply()
    t = transforms.NormalizeSphere()
    d.apply(t)

    # Test filter
    d.filter(lambda g: g.n_nodes >= 100)

    # Test map
    t = lambda g: g.n_nodes
    d.map(t, reduce=max)
