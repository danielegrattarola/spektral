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
from __future__ import absolute_import

import os

import networkx as nx
import numpy as np
import scipy.sparse as sp

from spektral.utils.io import load_binary

DATA_PATH = os.path.expanduser('~/.spektral/datasets/')
AVAILABLE_DATASETS = {'cora', 'citeseer', 'pubmed'}
RETURN_TYPES = {'masked', 'split'}


def _parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_name='cora', return_type='masked', val_size=500):
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError('Available datasets: {}'.format(AVAILABLE_DATASETS))
    if return_type not in RETURN_TYPES:
        raise ValueError('Possible return_type: {}'.format(RETURN_TYPES))

    if not os.path.exists(DATA_PATH + dataset_name):
        # TODO dataset downloader
        raise ValueError('Dataset not available. Download the data from '
                         'https://github.com/tkipf/gcn/tree/master/gcn/data and '
                         'place it in {}/'.format(DATA_PATH + dataset_name))

    print('Loading {} dataset'.format(dataset_name))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_path = os.path.join(DATA_PATH, dataset_name)
    for n in names:
        filename = "{}/ind.{}.{}".format(data_path, dataset_name, n)
        objects.append(load_binary(filename))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))

    if return_type is 'split':
        return adj, x, tx, allx, y, ty, ally
    else:
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(data_path, dataset_name))
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

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
