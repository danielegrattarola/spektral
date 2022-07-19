import os
import os.path as osp
import sys
import ssl
import errno
import urllib
import json

import numpy as np
import scipy.sparse as sp

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot


class Flickr(Dataset):
    """
    The Flickr dataset from the [Zeng at al. (2019)](https://arxiv.org/abs/1907.04931) paper, 
    containing descriptions and common properties of images.
    
    **Arguments**
    
    - `normalize_x`: if True, normalize the features.
    - `dtype`: numpy dtype of graph data.
    """

    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

    def __init__(self, normalize_x=False, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.normalize_x = normalize_x
        super().__init__(**kwargs)

    def download(self):
        print("Downloading Flickr dataset.")
        file_path = _download_url(self.url.format(self.adj_full_id), self.path)
        os.rename(file_path, osp.join(self.path, 'adj_full.npz'))

        file_path = _download_url(self.url.format(self.feats_id), self.path)
        os.rename(file_path, osp.join(self.path, 'feats.npy'))

        file_path = _download_url(self.url.format(self.class_map_id), self.path)
        os.rename(file_path, osp.join(self.path, 'class_map.json'))

        file_path = _download_url(self.url.format(self.role_id), self.path)
        os.rename(file_path, osp.join(self.path, 'role.json'))

    def read(self):
        f = np.load(osp.join(self.path, 'adj_full.npz'))
        a = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])

        x = np.load(osp.join(self.path, 'feats.npy'))

        if self.normalize_x:
            print("Pre-processing node features")
            x = _preprocess_features(x)

        y = np.zeros(x.shape[0])
        with open(osp.join(self.path, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                y[int(key)] = item

        y = label_to_one_hot(y, np.unique(y))

        with open(osp.join(self.path, 'role.json')) as f:
            role = json.load(f)

        self.train_mask = np.zeros(x.shape[0], dtype=bool)
        self.train_mask[np.array(role['tr'])] = 1

        self.val_mask = np.zeros(x.shape[0], dtype=bool)
        self.val_mask[np.array(role['va'])] = 1

        self.test_mask = np.zeros(x.shape[0], dtype=bool)
        self.test_mask[np.array(role['te'])] = 1

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]


def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def _download_url(url, folder, log=False):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]
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

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
