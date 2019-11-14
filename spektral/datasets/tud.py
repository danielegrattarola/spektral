import os
import shutil
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spektral.utils import nx_to_numpy

DATASET_URL = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
DATASET_CLEAN_URL = 'https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets'
DATA_PATH = os.path.expanduser('~/.spektral/datasets/')
AVAILABLE_DATASETS = [
    d[:-4]
    for d in pd.read_html(DATASET_URL)[0].Name[2:-1].values.tolist()
]


def load_data(dataset_name, normalize_features='zscore', clean=False):
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
    :param normalize_features: `'zscore'` or `'ohe'`, how to normalize the node
    features (only works for node attributes and node degrees).
    :param clean: if True, return a version of the dataset with no isomorphic
    graphs.
    :return:
    - a list of adjacency matrices;
    - a list of node feature matrices;
    - a numpy array containing the one-hot encoded targets.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError('Available datasets: {}'.format(AVAILABLE_DATASETS))

    if clean:
        dataset_name += '_clean'
    if not os.path.exists(DATA_PATH + dataset_name):
        _download_data(dataset_name)

    # Read data
    nx_graphs, y = _read_graphs(dataset_name)

    # Preprocessing
    y = np.array(y)[..., None]
    y = OneHotEncoder(sparse=False, categories='auto').fit_transform(y)

    # Get node attributes
    try:
        A, X_attr, _ = nx_to_numpy(nx_graphs, nf_keys=['attributes'], auto_pad=False)
        X_attr = _normalize_node_features(X_attr, normalize_features)
    except KeyError:
        print('Featureless nodes')
        A, X_attr, _ = nx_to_numpy(nx_graphs, auto_pad=False)  # na will be None

    # Get clustering coefficients (always zscore norm)
    clustering_coefficients = [np.array(list(nx.clustering(g).values()))[..., None] for g in nx_graphs]
    clustering_coefficients = _normalize_node_features(clustering_coefficients, 'zscore')

    # Get node degrees
    node_degrees = np.array([np.sum(_, axis=-1, keepdims=True) for _ in A])
    node_degrees = _normalize_node_features(node_degrees, normalize_features)

    # Get node labels (always ohe norm)
    try:
        _, X_labs, _ = nx_to_numpy(nx_graphs, nf_keys=['label'], auto_pad=False)
        X_labs = _normalize_node_features(X_labs, 'ohe')
    except KeyError:
        print('Label-less nodes')
        X_labs = None

    # Concatenate features
    Xs = [node_degrees, clustering_coefficients]
    if X_attr is not None:
        Xs.append(X_attr)
    if X_labs is not None:
        Xs.append(X_labs)
    X = [np.concatenate(x_, axis=-1) for x_ in zip(*Xs)]
    X = np.array(X)

    return A, X, y


def _read_graphs(dataset_name):
    file_prefix = DATA_PATH + dataset_name + '/' + dataset_name
    with open(file_prefix + "_graph_indicator.txt", "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]

    # Nodes
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_list = []
    vertex_list = []
    for i in node_indices:
        g = nx.Graph(directed=False)
        vertex_list_g = []
        for j in range(i[1] - i[0] + 1):
            vertex_list_g.append(g.add_node(j))

        graph_list.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open(file_prefix + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_list[g_id]
        off = offset[g_id]

        # Avoid multigraph
        edge_list.append(g.add_edge(e[0] - off, e[1] - off))

    # Node labels
    if os.path.exists(file_prefix + "_node_labels.txt"):
        with open(file_prefix + "_node_labels.txt", "r") as f:
            node_labels = [int(i) for i in list(f)]

        i = 0
        for g in graph_list:
            for n in g.nodes():
                g.nodes[n]['label'] = node_labels[i]
                i += 1

    # Node Attributes
    if os.path.exists(file_prefix + "_node_attributes.txt"):
        with open(file_prefix + "_node_attributes.txt", "r") as f:
            node_attributes = [map(float, i.split(',')) for i in list(f)]
        i = 0
        for g in graph_list:
            for n in g.nodes():
                g.nodes[n]['attributes'] = list(node_attributes[i])
                i += 1

    # Classes
    with open(file_prefix + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]

    return graph_list, classes


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
    with open(DATA_PATH + dataset_name + '.zip', 'wb') as out_file:
        out_file.write(req.content)
    with zipfile.ZipFile(DATA_PATH + dataset_name + '.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH + dataset_name + '/')
    os.remove(DATA_PATH + dataset_name + '.zip')

    subfolder = os.path.join(DATA_PATH, dataset_name, true_name)
    parentfolder = os.path.join(DATA_PATH, dataset_name)
    for filename in os.listdir(subfolder):
        try:
            suffix = filename.split(true_name)[1]
        except IndexError:
            # Probably the README
            continue
        shutil.move(
            os.path.join(subfolder, filename),
            os.path.join(parentfolder, dataset_name + suffix)
        )
    shutil.rmtree(subfolder)


def _normalize_node_features(feat_list, norm='ohe'):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == 'ohe':
        fnorm = OneHotEncoder(sparse=False, categories='auto')
    elif norm == 'zscore':
        fnorm = StandardScaler()
    else:
        raise ValueError('Possible feat_norm: ohe, zscore')
    fnorm.fit(np.vstack(feat_list))
    feat_list = [fnorm.transform(feat_.astype(np.float32)) for feat_ in feat_list]
    return feat_list
