import numpy as np
import scipy.sparse as sp

from spektral.data import Dataset, Graph
from spektral.utils import sparse


class OGB(Dataset):
    """
    Wrapper for datasets from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/).

    **Arguments**

    - `dataset`: an OGB library-agnostic dataset.

    """

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super().__init__(**kwargs)

    def read(self):
        if len(self.dataset) > 1:
            return [Graph(*_elem_to_numpy(elem)) for elem in self.dataset]
        else:
            # OGB crashed if we try to iterate over a NodePropPredDataset
            return [Graph(*_elem_to_numpy(self.dataset[0]))]


def _elem_to_numpy(elem):
    graph, label = elem
    n = graph["num_nodes"]
    x = graph["node_feat"]
    row, col = graph["edge_index"]
    e = graph["edge_feat"]
    a_e = sparse.edge_index_to_matrix(
        edge_index=np.array((row, col)).T,
        edge_weight=np.ones_like(row),
        edge_features=e,
        shape=(n, n),
    )
    if e is None:
        a = a_e
    else:
        a, e = a_e

    return x, a, e, label
