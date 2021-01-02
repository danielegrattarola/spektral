import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tensorflow.keras.utils import get_file
from tqdm import tqdm

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot
from spektral.utils.io import load_csv, load_sdf

ATOM_TYPES = [1, 6, 7, 8, 9]
BOND_TYPES = [1, 2, 3, 4]


class QM9(Dataset):
    """
    The QM9 chemical data set of small molecules.

    In this dataset, nodes represent atoms and edges represent chemical bonds.
    There are 5 possible atom types (H, C, N, O, F) and 4 bond types (single,
    double, triple, aromatic).

    Node features represent the chemical properties of each atom and include:

    - The atomic number, one-hot encoded;
    - The atom's position in the X, Y, and Z dimensions;
    - The atomic charge;
    - The mass difference from the monoisotope;

    The edge features represent the type of chemical bond between two atoms,
    one-hot encoded.

    Each graph has an 19-dimensional label for regression.

    **Arguments**

    - `amount`: int, load this many molecules instead of the full dataset
    (useful for debugging).
    - `n_jobs`: number of CPU cores to use for reading the data (-1, to use all
    available cores).
    """

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

    def __init__(self, amount=None, n_jobs=1, **kwargs):
        self.amount = amount
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    def download(self):
        get_file(
            "qm9.tar.gz",
            self.url,
            extract=True,
            cache_dir=self.path,
            cache_subdir=self.path,
        )
        os.remove(osp.join(self.path, "qm9.tar.gz"))

    def read(self):
        print("Loading QM9 dataset.")
        sdf_file = osp.join(self.path, "gdb9.sdf")
        data = load_sdf(sdf_file, amount=self.amount)  # Internal SDF format

        def read_mol(mol):
            x = np.array([atom_to_feature(atom) for atom in mol["atoms"]])
            a, e = mol_to_adj(mol)
            return x, a, e

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(read_mol)(mol) for mol in tqdm(data, ncols=80)
        )
        x_list, a_list, e_list = list(zip(*data))

        # Load labels
        labels_file = osp.join(self.path, "gdb9.sdf.csv")
        labels = load_csv(labels_file)
        labels = labels.set_index("mol_id").values
        if self.amount is not None:
            labels = labels[: self.amount]

        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]


def atom_to_feature(atom):
    atomic_num = label_to_one_hot(atom["atomic_num"], ATOM_TYPES)
    coords = atom["coords"]
    charge = atom["charge"]
    iso = atom["iso"]

    return np.concatenate((atomic_num, coords, [charge, iso]), -1)


def mol_to_adj(mol):
    row, col, edge_attr = [], [], []
    for bond in mol["bonds"]:
        start, end = bond["start_atom"], bond["end_atom"]
        row += [start, end]
        col += [end, start]
        edge_attr += [bond["type"]] * 2

    a = sp.csr_matrix((np.ones_like(row), (row, col)))
    edge_attr = np.array([label_to_one_hot(e, BOND_TYPES) for e in edge_attr])
    return a, edge_attr
