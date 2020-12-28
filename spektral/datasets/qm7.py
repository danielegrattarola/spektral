import os.path as osp

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from tensorflow.keras.utils import get_file

from spektral.data import Dataset, Graph


class QM7(Dataset):
    """
    The QM7b dataset of molecules from the paper:

    > [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564)<br>
    > Zhenqin Wu et al.

    The dataset has no node features.
    Edges and edge features are obtained from the Coulomb matrices of the
    molecules.

    Each graph has a 14-dimensional label for regression.
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self):
        get_file(
            "qm7b.mat",
            self.url,
            extract=True,
            cache_dir=self.path,
            cache_subdir=self.path,
        )

    def read(self):
        print("Loading QM7 dataset.")
        mat_file = osp.join(self.path, "qm7b.mat")
        data = loadmat(mat_file)

        coulomb_matrices = data["X"]
        labels = data["T"]

        output = []
        for i in range(len(coulomb_matrices)):
            row, col, data = sp.find(coulomb_matrices[i])
            a = sp.csr_matrix((np.ones_like(data), (row, col)))
            e = data[:, None]
            y = labels[i]
            output.append(Graph(a=a, e=e, y=y))

        return output
