import numpy as np

from spektral.data import Dataset, Graph
from spektral.data.utils import to_disjoint, to_batch, batch_generator
import scipy.sparse as sp
import numpy as np

ns = np.random.randint(3, 10, 10)
f = 3
a_list = [sp.csr_matrix(np.ones((n, n))) for n in ns]
x_list = [np.random.rand(n, f) for n in ns]
y = [[0, 1]] * len(ns)


def test_to_batch():
    # TODO test e_list
    x, a = to_batch(x_list, a_list)
    assert x.ndim == 3
    assert a.ndim == 3
    assert x.shape[0] == a.shape[0]
    assert x.shape[1] == a.shape[1] == a.shape[2]


def test_to_disjoint():
    # TODO test e_list
    x, a, i = to_disjoint(x_list, a_list)
    assert x.ndim == 2
    assert a.ndim == 2
    assert x.shape[0] == a.shape[0] == a.shape[1]


def test_batch_generator():
    size = 10
    batch_size = 6
    a = list(range(size))
    b = np.arange(size)

    class TestDataset(Dataset):
        def read(self):
            return [
                Graph(x=np.random.rand(n, 2),
                      adj=np.random.randint(0, 2, (n, n)),
                      y=np.array([0., 1.]))
                for n in range(size)
            ]

    c = TestDataset()

    batches = batch_generator([a, b, c], batch_size=batch_size, epochs=10)
    for batch in batches:
        a_, b_, c_ = batch
        for i in range(len(a_)):
            assert a_[i] == b_[i] == c_[i].N
