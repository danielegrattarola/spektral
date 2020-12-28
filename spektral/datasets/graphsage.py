import json
import os
import os.path as osp
import shutil
import zipfile

import numpy as np
import requests
import scipy.sparse as sp
from networkx.readwrite import json_graph

from spektral.data import Dataset, Graph
from spektral.data.dataset import DATASET_FOLDER
from spektral.datasets.utils import download_file


class GraphSage(Dataset):
    """
    The datasets used in the paper

    > [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)<br>
    > William L. Hamilton et al.

    The PPI dataset (originally
    [Stark et al. (2006)](https://www.ncbi.nlm.nih.gov/pubmed/16381927))
    for inductive node classification uses positional gene sets, motif gene sets
    and immunological signatures as features and gene ontology sets as labels.

    The Reddit dataset consists of a graph made of Reddit posts in the month of
    September, 2014. The label for each node is the community that a
    post belongs to. The graph is built by sampling 50 large communities and
    two nodes are connected if the same user commented on both. Node features
    are obtained by concatenating the average GloVe CommonCrawl vectors of
    the title and comments, the post's score and the number of comments.

    The train, test, and validation splits are given as binary masks and are
    accessible via the `mask_tr`, `mask_va`, and `mask_te` attributes.

    **Arguments**

    - `name`: name of the dataset to load (`'ppi'`, or `'reddit'`);
    """

    # TODO normalize features?
    # # # Z-score on features (optional)
    # if normalize_features:
    #     from sklearn.preprocessing import StandardScaler
    #     train_ids = np.array([id_map[n] for n in G.nodes()
    #                           if not G.nodes[n]['val'] and not G.nodes[n]['test']])
    #     x_tr = x[train_ids]
    #     scaler = StandardScaler()
    #     scaler.fit(x_tr)
    #     x = scaler.transform(x)

    url = "http://snap.stanford.edu/graphsage/{}.zip"

    def __init__(self, name, **kwargs):
        if name.lower() not in self.available_datasets:
            raise ValueError(
                "Unknown dataset: {}. Possible: {}".format(
                    name, self.available_datasets
                )
            )
        self.name = name.lower()
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, "GraphSage", self.name)

    def read(self):
        npz_file = osp.join(self.path, self.name) + ".npz"
        data = np.load(npz_file)
        x = data["x"]
        a = sp.csr_matrix(
            (data["adj_data"], (data["adj_row"], data["adj_col"])),
            shape=data["adj_shape"],
        )
        y = data["y"]
        self.mask_tr = data["mask_tr"]
        self.mask_va = data["mask_va"]
        self.mask_te = data["mask_te"]

        return [Graph(x=x, a=a, y=y)]

    def download(self):
        print("Downloading {} dataset.".format(self.name))
        url = self.url.format(self.name)
        download_file(url, self.path, self.name + ".zip")

        # Datasets are zipped in a folder: unpack them
        parent = self.path
        subfolder = osp.join(self.path, self.name)
        for filename in os.listdir(subfolder):
            shutil.move(osp.join(subfolder, filename), osp.join(parent, filename))
        os.rmdir(subfolder)

        x, adj, y, mask_tr, mask_va, mask_te = preprocess_data(self.path, self.name)

        # Save pre-processed data
        npz_file = osp.join(self.path, self.name) + ".npz"
        adj = adj.tocoo()
        np.savez(
            npz_file,
            x=x,
            adj_data=adj.data,
            adj_row=adj.row,
            adj_col=adj.col,
            adj_shape=adj.shape,
            y=y,
            mask_tr=mask_tr,
            mask_va=mask_va,
            mask_te=mask_te,
        )

    @property
    def available_datasets(self):
        return ["ppi", "reddit"]


class PPI(GraphSage):
    """
    Alias for `GraphSage('ppi')`.
    """

    def __init__(self, **kwargs):
        super().__init__(name="ppi", **kwargs)


class Reddit(GraphSage):
    """
    Alias for `GraphSage('reddit')`.
    """

    def __init__(self, **kwargs):
        super().__init__(name="reddit", **kwargs)


def preprocess_data(path, name):
    """
    Code adapted from https://github.com/williamleif/GraphSAGE
    """
    print("Processing dataset.")
    prefix = osp.join(path, name)

    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)

    x = np.load(prefix + "-feats.npy").astype(np.float32)

    id_map = json.load(open(prefix + "-id_map.json"))
    if list(id_map.keys())[0].isdigit():
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    n = len(id_map)

    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    [G.remove_node(node) for node in G.nodes() if node not in id_map]

    # Adjacency matrix
    edges = [
        (id_map[edge[0]], id_map[edge[1]])
        for edge in G.edges()
        if edge[0] in id_map and edge[1] in id_map
    ]
    edges = np.array(edges, dtype=np.int32)
    adj = sp.csr_matrix(
        (np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(n, n),
    )
    adj = adj.maximum(adj.transpose())

    # Process labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        y = np.zeros((n, num_classes), dtype=np.float32)
        for k in class_map.keys():
            y[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        y = np.zeros((n, num_classes), dtype=np.float32)
        for k in class_map.keys():
            y[id_map[k], class_map[k]] = 1

    # Get train/val/test indexes
    idx_va = np.array(
        [id_map[n] for n in G.nodes() if G.nodes[n]["val"]], dtype=np.int32
    )
    idx_te = np.array(
        [id_map[n] for n in G.nodes() if G.nodes[n]["test"]], dtype=np.int32
    )
    mask_tr = np.ones(n, dtype=np.bool)
    mask_va = np.zeros(n, dtype=np.bool)
    mask_te = np.zeros(n, dtype=np.bool)
    mask_tr[idx_va] = False
    mask_tr[idx_te] = False
    mask_va[idx_va] = True
    mask_te[idx_te] = True

    return x, adj, y, mask_tr, mask_va, mask_te
