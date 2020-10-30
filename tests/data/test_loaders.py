import numpy as np

from spektral.data import DisjointLoader, BatchLoader
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph
from spektral.data.loaders import PackedBatchLoader

n_graphs = 10
ns = np.random.randint(3, 8, n_graphs)
f = 3
s = 3
batch_size = 6

# batch size does not fit an integer number of times in n_graphs
graphs_in_batch = n_graphs % batch_size
assert graphs_in_batch != 0


class TestDataset(Dataset):
    def read(self):
        return [
            Graph(x=np.random.rand(n, f),
                  adj=np.random.randint(0, 2, (n, n)),
                  edge_attr=np.random.rand(n, n, s),
                  y=np.array([0., 1.]))
            for n in ns
        ]


def test_disjoint():
    data = TestDataset()
    loader = DisjointLoader(data, batch_size=batch_size)
    batches = [b for b in loader]

    (x, a, e, i), y = batches[-1]
    n = sum(ns[-graphs_in_batch:])
    assert x.shape == (n, f)
    assert a.shape == (n, n)
    assert len(e.shape) == 2 and e.shape[1] == s  # Avoid counting edges
    assert i.shape == (n, )
    assert y.shape == (graphs_in_batch, 2)


def test_batch():
    data = TestDataset()
    loader = BatchLoader(data, batch_size=batch_size)
    batches = [b for b in loader]

    (x, a, e), y = batches[-1]
    n = max(ns[-graphs_in_batch:])
    assert x.shape == (graphs_in_batch, n, f)
    assert a.shape == (graphs_in_batch, n, n)
    assert e.shape == (graphs_in_batch, n, n, s)
    assert y.shape == (graphs_in_batch, 2)


def test_fast_batch():
    data = TestDataset()
    loader = PackedBatchLoader(data, batch_size=batch_size)
    batches = [b for b in loader]

    (x, a, e), y = batches[-1]
    n = max(ns)
    assert x.shape == (graphs_in_batch, n, f)
    assert a.shape == (graphs_in_batch, n, n)
    assert e.shape == (graphs_in_batch, n, n, s)
    assert y.shape == (graphs_in_batch, 2)
