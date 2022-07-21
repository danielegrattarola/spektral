import errno
import os
import os.path as osp
import ssl
import sys
import urllib

import numpy as np
import scipy.sparse as sp

from spektral.data import Dataset, Graph
from spektral.datasets.citation import _preprocess_features
from spektral.datasets.utils import DATASET_FOLDER
from spektral.utils import label_to_one_hot


class DBLP(Dataset):
    """
    A subset of the DBLP computer science bibliography website,
    as collected in the
    [Fu et al. (2020)](https://arxiv.org/abs/2002.01680) paper.

    **Arguments**

    - `normalize_x`: if True, normalize the features.
    - `dtype`: numpy dtype of graph data.
    """

    def __init__(self, normalize_x=False, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.normalize_x = normalize_x
        super().__init__(**kwargs)

    url = "https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz"

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, "Citation", "dblp")

    def download(self):
        print("Downloading DBLP dataset.")
        _download_url(self.url, self.path)

    def read(self):
        f = np.load(osp.join(self.path, "dblp.npz"))

        x = sp.csr_matrix(
            (f["attr_data"], f["attr_indices"], f["attr_indptr"]), f["attr_shape"]
        ).toarray()
        x[x > 0] = 1

        if self.normalize_x:
            print("Pre-processing node features")
            x = _preprocess_features(x)

        a = sp.csr_matrix(
            (f["adj_data"], f["adj_indices"], f["adj_indptr"]), f["adj_shape"]
        )  # .tocoo()

        y = f["labels"]
        y = label_to_one_hot(y, np.unique(y))

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]


def _download_url(url, folder, log=False):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2]
    filename = filename if filename[0] == "?" else filename.split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log:
        print(f"Downloading {url}", file=sys.stderr)

    try:
        os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path
