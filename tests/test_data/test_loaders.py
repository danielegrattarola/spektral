import numpy as np
import scipy.sparse as sp

from spektral.data import BatchLoader, DisjointLoader
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph
from spektral.data.loaders import MixedLoader, PackedBatchLoader, SingleLoader

n_graphs = 10
ns = np.random.randint(3, 8, n_graphs)
f = 3
s = 3
batch_size = 6

# batch size does not fit an integer number of times in n_graphs
graphs_in_batch = n_graphs % batch_size
assert graphs_in_batch != 0


class TestDatasetSingle(Dataset):
    """
    A dataset with a single graph.
    """

    def read(self):
        n = 10
        return [
            Graph(
                x=np.random.rand(n, f),
                a=sp.csr_matrix(np.random.randint(0, 2, (n, n))),
                e=np.random.rand(n, n, s),
                y=np.array(n * [[0.0, 1.0]]),
            )
        ]


class TestDataset(Dataset):
    """
    A dataset with many graphs and graph-level labels
    """

    def read(self):
        return [
            Graph(
                x=np.random.rand(n, f),
                a=sp.csr_matrix(np.random.randint(0, 2, (n, n))),
                e=np.random.rand(n, n, s),
                y=np.array([0.0, 1.0]),
            )
            for n in ns
        ]


class TestDatasetDsjNode(Dataset):
    """
    A dataset with many graphs and node-level labels
    """

    def read(self):
        return [
            Graph(
                x=np.random.rand(n, f),
                a=sp.csr_matrix(np.random.randint(0, 2, (n, n))),
                e=np.random.rand(n, n, s),
                y=np.ones((n, 2)),
            )
            for n in ns
        ]


class TestDatasetMixed(Dataset):
    """
    A dataset in mixed mode
    """

    def read(self):
        n = np.random.randint(3, 8)
        self.a = sp.csr_matrix(np.random.randint(0, 2, (n, n)))
        return [
            Graph(
                x=np.random.rand(n, f),
                e=np.random.rand(n, n, s),
                y=np.array([0.0, 1.0]),
            )
            for _ in range(n_graphs)
        ]


def test_single():
    data = TestDatasetSingle()
    n = data.n_nodes
    loader = SingleLoader(data, sample_weights=np.ones(n), epochs=1)
    batches = list(loader)
    assert len(batches) == 1

    (x, a, e), y, sw = batches[0]
    assert x.shape == (n, f)
    assert a.shape == (n, n)
    assert len(e.shape) == 3 and e.shape[-1] == s  # Avoid counting edges
    assert y.shape == (n, 2)
    assert loader.steps_per_epoch == 1
    signature = loader.tf_signature()
    assert len(signature[0]) == 3


def test_disjoint():
    data = TestDataset()
    loader = DisjointLoader(data, batch_size=batch_size, epochs=1, shuffle=False)
    batches = list(loader)

    (x, a, e, i), y = batches[-1]
    n = sum(ns[-graphs_in_batch:])
    assert x.shape == (n, f)
    assert a.shape == (n, n)
    assert len(e.shape) == 2 and e.shape[1] == s  # Avoid counting edges
    assert i.shape == (n,)
    assert y.shape == (graphs_in_batch, 2)
    assert loader.steps_per_epoch == np.ceil(len(data) / batch_size)
    signature = loader.tf_signature()
    assert len(signature[0]) == 4


def test_disjoint_node():
    data = TestDatasetDsjNode()
    loader = DisjointLoader(
        data, node_level=True, batch_size=batch_size, epochs=1, shuffle=False
    )
    batches = list(loader)

    (x, a, e, i), y = batches[-1]
    n = sum(ns[-graphs_in_batch:])
    assert x.shape == (n, f)
    assert a.shape == (n, n)
    assert len(e.shape) == 2 and e.shape[1] == s  # Avoid counting edges
    assert i.shape == (n,)
    assert y.shape == (n, 2)
    assert loader.steps_per_epoch == np.ceil(len(data) / batch_size)

    signature = loader.tf_signature()
    assert len(signature[0]) == 4


def test_batch():
    data = TestDataset()
    loader = BatchLoader(data, batch_size=batch_size, epochs=1, shuffle=False)
    batches = list(loader)

    (x, a, e), y = batches[-1]
    n = max(ns[-graphs_in_batch:])
    assert x.shape == (graphs_in_batch, n, f)
    assert a.shape == (graphs_in_batch, n, n)
    assert e.shape == (graphs_in_batch, n, n, s)
    assert y.shape == (graphs_in_batch, 2)
    assert loader.steps_per_epoch == np.ceil(len(data) / batch_size)

    signature = loader.tf_signature()
    assert len(signature[0]) == 3


def test_packed_batch():
    data = TestDataset()
    loader = PackedBatchLoader(data, batch_size=batch_size, epochs=1, shuffle=False)
    batches = list(loader)

    (x, a, e), y = batches[-1]
    n = max(ns)
    assert x.shape == (graphs_in_batch, n, f)
    assert a.shape == (graphs_in_batch, n, n)
    assert e.shape == (graphs_in_batch, n, n, s)
    assert y.shape == (graphs_in_batch, 2)
    assert loader.steps_per_epoch == np.ceil(len(data) / batch_size)

    signature = loader.tf_signature()
    assert len(signature[0]) == 3


def test_mixed():
    data = TestDatasetMixed()
    loader = MixedLoader(data, batch_size=batch_size, epochs=1, shuffle=False)
    batches = list(loader)

    (x, a, e), y = batches[-1]
    n = data.n_nodes
    assert x.shape == (graphs_in_batch, n, f)
    assert a.shape == (n, n)
    assert e.shape == (graphs_in_batch, data.a.nnz, s)
    assert y.shape == (graphs_in_batch, 2)
    assert loader.steps_per_epoch == np.ceil(len(data) / batch_size)

    signature = loader.tf_signature()
    assert len(signature[0]) == 3
