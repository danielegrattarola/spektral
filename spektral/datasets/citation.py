"""
The MIT License

Copyright (c) 2016 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

This code was taken almost verbatim from https://github.com/tkipf/gcn/ and
adapted to work in Spektral.
"""
import os

import networkx as nx
import numpy as np
import requests
import scipy.sparse as sp

from spektral.utils.io import load_binary

DATA_URL = 'https://github.com/tkipf/gcn/raw/master/gcn/data/{}'
DATA_PATH = os.path.expanduser('~/.spektral/datasets/')
AVAILABLE_DATASETS = {'cora', 'citeseer', 'pubmed'}


def load_data(dataset_name='cora', normalize_features=True, random_split=False):
    """
    Loads a citation dataset (Cora, Citeseer or Pubmed) using the "Planetoid"
    splits intialliy defined in [Yang et al. (2016)](https://arxiv.org/abs/1603.08861).
    The train, test, and validation splits are given as binary masks.

    Node attributes are bag-of-words vectors representing the most common words
    in the text document associated to each node.
    Two papers are connected if either one cites the other.
    Labels represent the class of the paper.

    :param dataset_name: name of the dataset to load (`'cora'`, `'citeseer'`, or
    `'pubmed'`);
    :param normalize_features: if True, the node features are normalized;
    :param random_split: if True, return a randomized split (20 nodes per class
    for training, 30 nodes per class for validation and the remaining nodes for
    testing, [Shchur et al. (2018)](https://arxiv.org/abs/1811.05868)).
    :return:
        - Adjacency matrix;
        - Node features;
        - Labels;
        - Three binary masks for train, validation, and test splits.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError('Available datasets: {}'.format(AVAILABLE_DATASETS))

    if not os.path.exists(DATA_PATH + dataset_name):
        _download_data(dataset_name)

    print('Loading {} dataset'.format(dataset_name))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_path = os.path.join(DATA_PATH, dataset_name)
    for n in names:
        filename = os.path.join(data_path, 'ind.{}.{}'.format(dataset_name, n))
        objects.append(load_binary(filename))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = _parse_index_file(
        os.path.join(data_path, "ind.{}.test.index".format(dataset_name)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # Row-normalize the features
    if normalize_features:
        print('Pre-processing node features')
        features = _preprocess_features(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # Data splits
    if random_split:
        from sklearn.model_selection import train_test_split
        indices = np.arange(labels.shape[0])
        n_classes = labels.shape[1]
        idx_train, idx_test, y_train, y_test = train_test_split(
            indices, labels, train_size=20 * n_classes, stratify=labels)
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_test, y_test, train_size=30 * n_classes, stratify=y_test)
    else:
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    train_mask = _sample_mask(idx_train, labels.shape[0])
    val_mask = _sample_mask(idx_val, labels.shape[0])
    test_mask = _sample_mask(idx_test, labels.shape[0])

    return adj, features, labels, train_mask, val_mask, test_mask


def _parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def _download_data(dataset_name):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']

    os.makedirs(os.path.join(DATA_PATH, dataset_name))

    print('Downloading', dataset_name, 'from', DATA_URL[:-2])
    for n in names:
        f_name = 'ind.{}.{}'.format(dataset_name, n)
        req = requests.get(DATA_URL.format(f_name))

        with open(os.path.join(DATA_PATH, dataset_name, f_name), 'wb') as out_file:
            out_file.write(req.content)
