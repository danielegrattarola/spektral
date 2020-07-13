import glob
import os
import shutil
import zipfile
from os import path as osp
from urllib.error import URLError

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spektral.utils import io

DATASET_URL = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
DATASET_CLEAN_URL = 'https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets'
DATA_PATH = osp.expanduser('~/.spektral/datasets/')


def load_data(dataset_name, clean=False):
    """
    Loads one of the Benchmark Data Sets for Graph Kernels from TU Dortmund
    ([link](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).
    The node features are computed by concatenating the following features for
    each node:

    - node attributes, if available, normalized as specified in `normalize_features`;
    - clustering coefficient, normalized with z-score;
    - node degrees, normalized as specified in `normalize_features`;
    - node labels, if available, one-hot encoded.
    :param dataset_name: name of the dataset to load (see `spektral.datasets.tud.AVAILABLE_DATASETS`).
    :param clean: if True, return a version of the dataset with no isomorphic
    graphs.
    :return:
    - a list of adjacency matrices;
    - a list of node feature matrices;
    - a numpy array containing the one-hot encoded targets.
    """
    if clean:
        dataset_name += '_clean'
    if not osp.exists(DATA_PATH + dataset_name):
        _download_data(dataset_name)

    # Read data
    A_list, X_list, y = _read_graphs(dataset_name)

    print('Successfully loaded {}.'.format(dataset_name))

    return A_list, X_list, y


def available_datasets():
    try:
        return [
            d[:-4]
            for d in pd.read_html(DATASET_URL)[0].Name[2:-1].values.tolist()
        ]
    except URLError:
        # No internet, don't panic
        print('No connection. See {}'.format(DATASET_URL))
        return []


def _read_graphs(dataset_name):
    file_prefix = osp.join(DATA_PATH, dataset_name, dataset_name)
    available = [
        f.split(os.sep)[-1][len(dataset_name)+1:-4]
        for f in glob.glob('{}_*.txt'.format(file_prefix))
    ]

    I = io.load_txt(file_prefix + '_graph_indicator.txt').astype(int) - 1
    unique_ids = np.unique(I)
    num_graphs = len(unique_ids)
    graph_sizes = np.bincount(I)
    offsets = np.concatenate(([0], np.cumsum(graph_sizes)[:-1]))
    edges = io.load_txt(file_prefix + '_A.txt', delimiter=',').astype(int) - 1

    A_list = [[] for _ in range(num_graphs)]
    for e in edges:
        graph_id = I[e[0]]
        A_list[graph_id].append(e - offsets[graph_id])
    A_list = map(np.array, A_list)
    A_list = [
        sp.coo_matrix(
            (np.ones_like(A[:, 0]), (A[:, 0], A[:, 1])),
            shape=(graph_sizes[i], graph_sizes[i])
        )
        for i, A in enumerate(A_list)
    ]

    X = []
    if 'node_attributes' in available:
        X_na = io.load_txt(file_prefix + '_node_attributes.txt', delimiter=',')
        if X_na.ndim == 1:
            X_na = X_na[:, None]
        X.append(X_na)
    if 'node_labels' in available:
        X_nl = io.load_txt(file_prefix + '_node_labels.txt')
        X_nl = _normalize(X_nl.reshape(-1, 1), 'ohe')
        X.append(X_nl)
    if len(X) > 0:
        X = np.concatenate(X, -1)

    X_list = []
    start = offsets[0]
    for i in range(num_graphs):
        stop = offsets[i + 1] if i + 1 < len(offsets) else None
        X_list.append(X[start:stop])
        start = stop

    y = None
    if 'graph_attributes' in available:
        y = io.load_txt(file_prefix + '_graph_attributes.txt')
    elif 'graph_labels' in available:
        y = io.load_txt(file_prefix + '_graph_labels.txt')
        y = _normalize(y[:, None], 'ohe')

    return A_list, X_list, y


def _download_data(dataset_name):
    print('Dowloading ' + dataset_name + ' dataset.')
    if dataset_name.endswith('_clean'):
        true_name = dataset_name[:-6]
        url = DATASET_CLEAN_URL
    else:
        true_name = dataset_name
        url = DATASET_URL

    data_url = '{}/{}.zip'.format(url, true_name)
    req = requests.get(data_url)
    if req.status_code == 404:
        raise ValueError('Unknown dataset {}. See spektral.datasets.tud.available_datasets()'
                         ' for a list of available datasets.'
                         .format(dataset_name))

    os.makedirs(DATA_PATH, exist_ok=True)
    with open(DATA_PATH + dataset_name + '.zip', 'wb') as out_file:
        out_file.write(req.content)
    with zipfile.ZipFile(DATA_PATH + dataset_name + '.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH + dataset_name + '/')
    os.remove(DATA_PATH + dataset_name + '.zip')

    subfolder = osp.join(DATA_PATH, dataset_name, true_name)
    parentfolder = osp.join(DATA_PATH, dataset_name)
    for filename in os.listdir(subfolder):
        try:
            suffix = filename.split(true_name)[1]
        except IndexError:
            # Probably the README
            continue
        shutil.move(
            osp.join(subfolder, filename),
            osp.join(parentfolder, dataset_name + suffix)
        )
    shutil.rmtree(subfolder)


def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == 'ohe':
        fnorm = OneHotEncoder(sparse=False, categories='auto')
    elif norm == 'zscore':
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)
