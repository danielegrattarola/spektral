import os
import os.path as osp

import networkx as nx
import numpy as np
import requests
import scipy.sparse as sp

from spektral.data import Dataset, Graph
from spektral.datasets.utils import DATASET_FOLDER
from spektral.utils.io import load_binary


class Citation(Dataset):
    """
    The citation datasets Cora, Citeseer and Pubmed.

    Node attributes are bag-of-words vectors representing the most common words
    in the text document associated to each node.
    Two papers are connected if either one cites the other.
    Labels represent the subject area of the paper.

    The train, test, and validation splits are given as binary masks and are
    accessible via the `mask_tr`, `mask_va`, and `mask_te` attributes.

    **Arguments**

    - `name`: name of the dataset to load (`'cora'`, `'citeseer'`, or
    `'pubmed'`);
    - `random_split`: if True, return a randomized split (20 nodes per class
    for training, 30 nodes per class for validation and the remaining nodes for
    testing, as recommended by [Shchur et al. (2018)](https://arxiv.org/abs/1811.05868)).
    If False (default), return the "Planetoid" public splits defined by
    [Yang et al. (2016)](https://arxiv.org/abs/1603.08861).
    - `normalize_x`: if True, normalize the features.
    - `dtype`: numpy dtype of graph data.
    """

    url = "https://github.com/tkipf/gcn/raw/master/gcn/data/{}"
    suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]

    def __init__(
        self, name, random_split=False, normalize_x=False, dtype=np.float32, **kwargs
    ):
        if hasattr(dtype, "as_numpy_dtype"):
            # support tf.dtypes
            dtype = dtype.as_numpy_dtype
        self.name = name.lower()
        if self.name not in self.available_datasets:
            raise ValueError(
                "Unknown dataset {}. See Citation.available_datasets "
                "for a list of available datasets."
            )
        self.random_split = random_split
        self.normalize_x = normalize_x
        self.mask_tr = self.mask_va = self.mask_te = None
        self.dtype = dtype
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, "Citation", self.name)

    def read(self):
        objects = [_read_file(self.path, self.name, s) for s in self.suffixes]
        objects = [o.A if sp.issparse(o) else o for o in objects]
        x, y, tx, ty, allx, ally, graph, idx_te = objects

        # Public Planetoid splits. This is the default
        idx_tr = np.arange(y.shape[0])
        idx_va = np.arange(y.shape[0], y.shape[0] + 500)
        idx_te = idx_te.astype(int)
        idx_te_sort = np.sort(idx_te)

        # Fix disconnected nodes in Citeseer
        if self.name == "citeseer":
            idx_te_len = idx_te.max() - idx_te.min() + 1
            tx_ext = np.zeros((idx_te_len, x.shape[1]))
            tx_ext[idx_te_sort - idx_te.min(), :] = tx
            tx = tx_ext
            ty_ext = np.zeros((idx_te_len, y.shape[1]))
            ty_ext[idx_te_sort - idx_te.min(), :] = ty
            ty = ty_ext

        x = np.vstack((allx, tx))
        y = np.vstack((ally, ty))
        x[idx_te, :] = x[idx_te_sort, :]
        y[idx_te, :] = y[idx_te_sort, :]

        # Row-normalize the features
        if self.normalize_x:
            print("Pre-processing node features")
            x = _preprocess_features(x)

        if self.random_split:
            # Throw away public splits and compute random ones like Shchur et al.
            from sklearn.model_selection import train_test_split

            indices = np.arange(y.shape[0])
            n_classes = y.shape[1]
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                indices, y, train_size=20 * n_classes, stratify=y
            )
            idx_va, idx_te, y_va, y_te = train_test_split(
                idx_te, y_te, train_size=30 * n_classes, stratify=y_te
            )

        # Adjacency matrix
        a = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # CSR
        a.setdiag(0)
        a.eliminate_zeros()

        # Train/valid/test masks
        self.mask_tr = _idx_to_mask(idx_tr, y.shape[0])
        self.mask_va = _idx_to_mask(idx_va, y.shape[0])
        self.mask_te = _idx_to_mask(idx_te, y.shape[0])

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]

    def download(self):
        print("Downloading {} dataset.".format(self.name))
        os.makedirs(self.path, exist_ok=True)
        for n in self.suffixes:
            f_name = "ind.{}.{}".format(self.name, n)
            req = requests.get(self.url.format(f_name))
            if req.status_code == 404:
                raise ValueError(
                    "Cannot download dataset ({} returned 404).".format(
                        self.url.format(f_name)
                    )
                )
            with open(os.path.join(self.path, f_name), "wb") as out_file:
                out_file.write(req.content)

    @property
    def available_datasets(self):
        return ["cora", "citeseer", "pubmed"]


class Cora(Citation):
    """
    Alias for `Citation('cora')`.
    """

    def __init__(self, random_split=False, normalize_x=False, **kwargs):
        super().__init__(
            "cora", random_split=random_split, normalize_x=normalize_x, **kwargs
        )


class Citeseer(Citation):
    """
    Alias for `Citation('citeseer')`.
    """

    def __init__(self, random_split=False, normalize_x=False, **kwargs):
        super().__init__(
            "citeseer", random_split=random_split, normalize_x=normalize_x, **kwargs
        )


class Pubmed(Citation):
    """
    Alias for `Citation('pubmed')`.
    """

    def __init__(self, random_split=False, normalize_x=False, **kwargs):
        super().__init__(
            "pubmed", random_split=random_split, normalize_x=normalize_x, **kwargs
        )


def _read_file(path, name, suffix):
    full_fname = os.path.join(path, "ind.{}.{}".format(name, suffix))
    if suffix == "test.index":
        return np.loadtxt(full_fname)
    else:
        return load_binary(full_fname)


def _idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
