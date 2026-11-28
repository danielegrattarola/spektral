import os
import pickle

import numpy as np
import scipy.sparse as sp

from spektral.datasets.citation import Citation


def _write_planetoid_files(directory, name="cora"):
    n_tr, n_va, n_te, n_feat, n_cls = 4, 500, 3, 5, 2
    n_allx = n_tr + n_va
    n_total = n_allx + n_te

    objects = {
        "x": sp.csr_matrix(np.eye(n_tr, n_feat, dtype=np.float32)),
        "y": np.eye(n_tr, n_cls, dtype=np.float32),
        "tx": sp.csr_matrix(np.ones((n_te, n_feat), dtype=np.float32)),
        "ty": np.ones((n_te, n_cls), dtype=np.float32),
        "allx": sp.csr_matrix(np.ones((n_allx, n_feat), dtype=np.float32)),
        "ally": np.ones((n_allx, n_cls), dtype=np.float32),
        "graph": {i: [(i + 1) % n_total] for i in range(n_total)},
    }
    for suffix, obj in objects.items():
        with open(os.path.join(directory, f"ind.{name}.{suffix}"), "wb") as f:
            pickle.dump(obj, f)

    idx_te = np.arange(n_allx, n_total)
    np.savetxt(os.path.join(directory, f"ind.{name}.test.index"), idx_te, fmt="%d")
    return n_total, n_feat, n_cls


def test_citation_read_sparse_objects(tmp_path):
    n_total, n_feat, n_cls = _write_planetoid_files(tmp_path, name="cora")

    class LocalCitation(Citation):
        @property
        def path(self):
            return str(tmp_path)

        def download(self):
            raise AssertionError("download should not be called")

    dataset = LocalCitation("cora")
    assert len(dataset) == 1
    graph = dataset[0]
    assert graph.x.shape == (n_total, n_feat)
    assert graph.a.shape == (n_total, n_total)
    assert graph.y.shape == (n_total, n_cls)
    assert dataset.mask_tr.sum() == 4
    assert dataset.mask_va.sum() == 500
    assert dataset.mask_te.sum() == 3
