import glob
import os
import shutil
from os import path as osp
from urllib.error import URLError

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spektral.data import Dataset, Graph
from spektral.datasets.utils import download_file
from spektral.utils import io, sparse


class TUDataset(Dataset):
    """
    The Benchmark Data Sets for Graph Kernels from TU Dortmund
    ([link](https://chrsmrrs.github.io/datasets/docs/datasets/)).

    Node features are computed by concatenating the following features for
    each node:

    - node attributes, if available;
    - node labels, if available, one-hot encoded.

    Some datasets might not have node features at all. In this case, attempting
    to use the dataset with a Loader will result in a crash. You can create
    node features using some of the transforms available in `spektral.transforms`
    or you can define your own features by accessing the individual samples in
    the `graph` attribute of the dataset (which is a list of `Graph` objects).

    Edge features are computed by concatenating the following features for
    each node:

    - edge attributes, if available;
    - edge labels, if available, one-hot encoded.

    Graph labels are provided for each dataset.

    Specific details about each individual dataset can be found in
    `~/.spektral/datasets/TUDataset/<dataset name>/README.md`, after the dataset
    has been downloaded locally (datasets are downloaded automatically upon
    calling `TUDataset('<dataset name>')` the first time).

    **Arguments**

    - `name`: str, name of the dataset to load (see `TUD.available_datasets`).
    - `clean`: if `True`, rload a version of the dataset with no isomorphic
               graphs.
    """

    url = "https://www.chrsmrrs.com/graphkerneldatasets"
    url_clean = (
        "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets"
    )

    def __init__(self, name, clean=False, **kwargs):
        if name not in self.available_datasets():
            raise ValueError(
                "Unknown dataset {}. See {}.available_datasets() for a complete list of"
                "available datasets.".format(name, self.__class__.__name__)
            )
        self.name = name
        self.clean = clean
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(super().path, self.name + ("_clean" if self.clean else ""))

    def download(self):
        print(
            "Downloading {} dataset{}.".format(
                self.name, " (clean)" if self.clean else ""
            )
        )
        url = "{}/{}.zip".format(self.url_clean if self.clean else self.url, self.name)
        download_file(url, self.path, self.name + ".zip")

        # Datasets are zipped in a folder: unpack them
        parent = self.path
        subfolder = osp.join(self.path, self.name)
        for filename in os.listdir(subfolder):
            shutil.move(osp.join(subfolder, filename), osp.join(parent, filename))
        os.rmdir(subfolder)

    def read(self):
        fname_template = osp.join(self.path, "{}_{{}}.txt".format(self.name))
        available = [
            f.split(os.sep)[-1][len(self.name) + 1 : -4]  # Remove leading name
            for f in glob.glob(fname_template.format("*"))
        ]

        # Batch index
        node_batch_index = (
            io.load_txt(fname_template.format("graph_indicator")).astype(int) - 1
        )
        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # Read edge lists
        edges = io.load_txt(fname_template.format("A"), delimiter=",").astype(int) - 1
        # Remove duplicates and self-loops from edges
        _, mask = np.unique(edges, axis=0, return_index=True)
        mask = mask[edges[mask, 0] != edges[mask, 1]]
        edges = edges[mask]
        # Split edges into separate edge lists
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(edges - n_nodes_cum[edge_batch_idx, None], n_edges_cum)

        # Node features
        x_list = []
        if "node_attributes" in available:
            x_attr = io.load_txt(
                fname_template.format("node_attributes"), delimiter=","
            )
            if x_attr.ndim == 1:
                x_attr = x_attr[:, None]
            x_list.append(x_attr)
        if "node_labels" in available:
            x_labs = io.load_txt(fname_template.format("node_labels"))
            if x_labs.ndim == 1:
                x_labs = x_labs[:, None]
            x_labs = np.concatenate(
                [_normalize(xl_[:, None], "ohe") for xl_ in x_labs.T], -1
            )
            x_list.append(x_labs)
        if len(x_list) > 0:
            x_list = np.concatenate(x_list, -1)
            x_list = np.split(x_list, n_nodes_cum[1:])
        else:
            print(
                "WARNING: this dataset doesn't have node attributes."
                "Consider creating manual features before using it with a "
                "Loader."
            )
            x_list = [None] * len(n_nodes)

        # Edge features
        e_list = []
        if "edge_attributes" in available:
            e_attr = io.load_txt(fname_template.format("edge_attributes"))
            if e_attr.ndim == 1:
                e_attr = e_attr[:, None]
            e_attr = e_attr[mask]
            e_list.append(e_attr)
        if "edge_labels" in available:
            e_labs = io.load_txt(fname_template.format("edge_labels"))
            if e_labs.ndim == 1:
                e_labs = e_labs[:, None]
            e_labs = e_labs[mask]
            e_labs = np.concatenate(
                [_normalize(el_[:, None], "ohe") for el_ in e_labs.T], -1
            )
            e_list.append(e_labs)
        if len(e_list) > 0:
            e_available = True
            e_list = np.concatenate(e_list, -1)
            e_list = np.split(e_list, n_edges_cum)
        else:
            e_available = False
            e_list = [None] * len(n_nodes)

        # Create sparse adjacency matrices and re-sort edge attributes in lexicographic
        # order
        a_e_list = [
            sparse.edge_index_to_matrix(
                edge_index=el,
                edge_weight=np.ones(el.shape[0]),
                edge_features=e,
                shape=(n, n),
            )
            for el, e, n in zip(el_list, e_list, n_nodes)
        ]
        if e_available:
            a_list, e_list = list(zip(*a_e_list))
        else:
            a_list = a_e_list

        # Labels
        if "graph_attributes" in available:
            labels = io.load_txt(fname_template.format("graph_attributes"))
        elif "graph_labels" in available:
            labels = io.load_txt(fname_template.format("graph_labels"))
            labels = _normalize(labels[:, None], "ohe")
        else:
            raise ValueError("No labels available for dataset {}".format(self.name))

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

    @staticmethod
    def available_datasets():
        url = "https://chrsmrrs.github.io/datasets/docs/datasets/"
        try:
            tables = pd.read_html(url)
            names = []
            for table in tables:
                names.extend(table.Name[1:].values.tolist())
            return names
        except URLError:
            # No internet, don't panic
            print("Could not read URL {}".format(url))
            return []


def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == "ohe":
        fnorm = OneHotEncoder(sparse=False, categories="auto")
    elif norm == "zscore":
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)
