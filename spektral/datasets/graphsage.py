"""
The MIT License

Copyright (c) 2017 William L. Hamilton, Rex Ying

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

Portions of this code base were orginally forked from: https://github.com/tkipf/gcn, which is under the following License:

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

Note from Spektral's authors: the code by Hamilton et al. was adapted and the
present version is not a verbatim copy.
"""

import json
import os
import zipfile

import numpy as np
import requests
import scipy.sparse as sp
from networkx.readwrite import json_graph

DATA_PATH = os.path.expanduser('~/.spektral/datasets/')
AVAILABLE_DATASETS = {'ppi', 'reddit'}


def load_data(dataset_name, max_degree=-1, normalize_features=True):
    """
    Loads one of the datasets (PPI or Reddit) used in
    [Hamilton & Ying (2017)](https://arxiv.org/abs/1706.02216).

    The PPI dataset (originally [Stark et al. (2006)](https://www.ncbi.nlm.nih.gov/pubmed/16381927))
    for inductive node classification uses positional gene sets, motif gene sets
    and immunological signatures as features and gene ontology sets as labels.

    The Reddit dataset consists of a graph made of Reddit posts in the month of
    September, 2014. The label for each node is the community that a
    post belongs to. The graph is built by sampling 50 large communities and
    two nodes are connected if the same user commented on both. Node features
    are obtained by concatenating the average GloVe CommonCrawl vectors of
    the title and comments, the post's score and the number of comments.

    The train, test, and validation splits are returned as binary masks.

    :param dataset_name: name of the dataset to load (`'ppi'`, or `'reddit'`);
    :param max_degree: int, if positive, subsample edges so that each node has
    the specified maximum degree.
    :param normalize_features: if True, the node features are normalized;
    :return:
        - Adjacency matrix;
        - Node features;
        - Labels;
        - Three binary masks for train, validation, and test splits.
    """
    prefix = DATA_PATH + dataset_name + '/' + dataset_name
    if max_degree == -1:
        npz_file = prefix + '.npz'
    else:
        npz_file = '{}_deg{}.npz'.format(prefix, max_degree)

    if not os.path.exists(prefix + "-G.json"):
        _download_data(dataset_name)

    if os.path.exists(npz_file):
        # Data already prepreccesed
        print('Loading pre-processed dataset {}.'.format(npz_file))
        data = np.load(npz_file)
        feats = data['feats']
        labels = data['labels']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
        full_adj = sp.csr_matrix(
            (data['full_adj_data'], data['full_adj_indices'], data['full_adj_indptr']),
            shape=data['full_adj_shape']
        )
    else:
        # Preprocess data
        print('Loading dataset.')
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
        feats = np.load(prefix + "-feats.npy").astype(np.float32)
        id_map = json.load(open(prefix + "-id_map.json"))
        if list(id_map.keys())[0].isdigit():
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        id_map = {conversion(k): int(v) for k, v in id_map.items()}
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n: n
        else:
            lab_conversion = lambda n: int(n)

        class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

        # Remove all nodes that do not have val/test annotations
        # (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        to_remove = []
        for node in G.nodes():
            if node not in id_map:
                to_remove.append(node)
                broken_count += 1
        for node in to_remove:
            G.remove_node(node)
        print(
            "Removed {:d} nodes that lacked proper annotations due to networkx versioning issues"
            .format(broken_count)
        )

        # Construct adjacency matrix
        edges = []
        for edge in G.edges():
            if edge[0] in id_map and edge[1] in id_map:
                edges.append((id_map[edge[0]], id_map[edge[1]]))
        print('{} edges'.format(len(edges)))
        num_data = len(id_map)

        # Subsample edges (optional)
        if max_degree > -1:
            print('Subsampling edges.')
            edges = _subsample_edges(edges, num_data, max_degree)

        # Get train/val/test indexes
        val_data = np.array([id_map[n] for n in G.nodes()
                             if G.nodes[n]['val']], dtype=np.int32)
        test_data = np.array([id_map[n] for n in G.nodes()
                              if G.nodes[n]['test']], dtype=np.int32)
        train_mask = np.ones((num_data), dtype=np.bool)
        train_mask[val_data] = False
        train_mask[test_data] = False
        val_mask = np.zeros((num_data), dtype=np.bool)
        val_mask[val_data] = True
        test_mask = np.zeros((num_data), dtype=np.bool)
        test_mask[test_data] = True

        edges = np.array(edges, dtype=np.int32)

        def _get_adj(edges):
            adj = sp.csr_matrix((np.ones((edges.shape[0]), dtype=np.float32),
                                 (edges[:, 0], edges[:, 1])), shape=(num_data, num_data))
            adj = adj.maximum(adj.transpose())
            return adj

        full_adj = _get_adj(edges)

        # Z-score on features (optional)
        if normalize_features:
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([id_map[n] for n in G.nodes()
                                  if not G.nodes[n]['val'] and not G.nodes[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)

        # Process labels
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], class_map[k]] = 1

        with open(npz_file, 'wb') as fwrite:
            print('Saving {} edges'.format(full_adj.nnz))
            np.savez(fwrite, num_data=num_data,
                     full_adj_data=full_adj.data, full_adj_indices=full_adj.indices, full_adj_indptr=full_adj.indptr,
                     full_adj_shape=full_adj.shape,
                     feats=feats,
                     labels=labels,
                     train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return full_adj, feats, labels, train_mask, val_mask, test_mask


def _download_data(dataset_name):
    print('Dowloading ' + dataset_name + ' dataset.')
    if dataset_name == 'ppi':
        data_url = 'http://snap.stanford.edu/graphsage/ppi.zip'
    elif dataset_name == 'reddit':
        data_url = 'http://snap.stanford.edu/graphsage/reddit.zip'
    else:
        raise ValueError('dataset_name must be one of: {}'.format(AVAILABLE_DATASETS))
    req = requests.get(data_url)

    os.makedirs(DATA_PATH, exist_ok=True)
    with open(DATA_PATH + dataset_name + '.zip', 'wb') as out_file:
        out_file.write(req.content)
    with zipfile.ZipFile(DATA_PATH + dataset_name + '.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)


def _subsample_edges(edges, num_data, max_degree):
    edges = np.array(edges, dtype=np.int32)
    np.random.shuffle(edges)
    degree = np.zeros(num_data, dtype=np.int32)

    new_edges = []
    for e in edges:
        if degree[e[0]] < max_degree and degree[e[1]] < max_degree:
            new_edges.append((e[0], e[1]))
            degree[e[0]] += 1
            degree[e[1]] += 1
    return new_edges
