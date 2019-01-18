from __future__ import absolute_import

import os
from glob import glob

import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

from spektral.utils.io import load_dot
from spektral.utils import natural_key, label_to_one_hot, numpy_to_nx, nx_to_numpy

DATA_PATH = os.path.expanduser('~/.spektral/datasets/delaunay/')
RETURN_TYPES = {'numpy', 'networkx'}
NODE_FEATURES = ['x', 'y']
NF_PROCESSING = [None, None]
MAX_K = 7  # Maximum number of nodes in a graph


def load_data(return_type='networkx', nf_keys=None, self_loops=False,
              one_hot_labels=True, classes=None):
    """
    Loads the Delaunay triangulations dataset by [Zambon et al. (2017)](https://arxiv.org/abs/1706.06941).
    The dataset is currently not publicly available for download, but you can
    request it via e-mail to D. Grattarola (daniele.gratttarola@gmail.com).
    :param return_type: `'networkx'` or `'numpy'`, data format to return;
    :param nf_keys: list or str, node features to return (see `delaunay.NODE_FEATURES`
    for available features);
    :param self_loops: if `return_type='numpy'`, add self loops to adjacency 
    matrices;
    :param one_hot_labels: one-hot encode dataset labels; 
    :param classes: indices of the classes to load (integer, or list of integers
    between 0 and 20).
    :return: if `return_type='networkx'`, a list of graphs in Networkx format, 
    and an array containing labels; if `return_type='numpy'`, the adjacency 
    matrix, node features, and an array containing labels.
    """
    if return_type not in RETURN_TYPES:
        raise ValueError('Possible return_type: {}'.format(RETURN_TYPES))
    if not os.path.exists(DATA_PATH):
        # TODO dataset downloader
        raise ValueError('Dataset not available. Download it and place it in {}'
                         ''.format(DATA_PATH))

    print('Loading Delaunay dataset.')

    n_classes = len(glob(DATA_PATH + '/*'))
    if classes is None:
        classes = list(range(n_classes))

    data = []
    labels = []
    for subdir_idx in tqdm(classes):
        subdir_wc = os.path.join(DATA_PATH, str(subdir_idx), '*.dot')
        files = sorted(glob(subdir_wc), key=natural_key)
        for f in files:
            data.append(load_dot(f))
            labels.append(subdir_idx)
    labels = np.array(labels)

    if one_hot_labels:
        labels = label_to_one_hot(labels, labels=classes)

    if nf_keys is not None:
        if isinstance(nf_keys, str):
            nf_keys = [nf_keys]
    else:
        nf_keys = NODE_FEATURES

    if return_type is 'numpy':
        adj, nf, _ = nx_to_numpy(data,
                                 auto_pad=False, self_loops=self_loops,
                                 nf_keys=nf_keys,
                                 nf_postprocessing=NF_PROCESSING)

        return adj, nf, labels
    elif return_type is 'networkx':
        return data, labels
    else:
        raise NotImplementedError


def generate_data(return_type='networkx', classes=0, n_samples_in_class=1000,
                  n_nodes=7, support_low=0., support_high=10., drift_amount=1.0,
                  one_hot_labels=True, support=None, seed=None):
    """
    Generates a dataset of Delaunay triangulations as described by
    [Zambon et al. (2017)](https://arxiv.org/abs/1706.06941).
    Note that this code is basically deprecated and will change soon.
    
    :param return_type: `'networkx'` or `'numpy'`, data format to return;
    :param classes: indices of the classes to load (integer, or list of integers
    between 0 and 20);
    :param n_samples_in_class: number of generated samples per class;
    :param n_nodes: number of nodes in a graph;
    :param support_low: lower bound of the uniform distribution from which the 
    support is generated;
    :param support_high: upper bound of the uniform distribution from which the 
    support is generated;
    :param drift_amount: coefficient to control the amount of change between 
    classes;
    :param one_hot_labels: one-hot encode dataset labels;
    :param support: custom support to use instead of generating it randomly; 
    :param seed: random numpy seed;
    :return: if `return_type='networkx'`, a list of graphs in Networkx format, 
    and an array containing labels; if `return_type='numpy'`, the adjacency 
    matrix, node features, and an array containing labels.
    """
    if return_type not in RETURN_TYPES:
        raise ValueError('Possible return_type: {}'.format(RETURN_TYPES))

    if isinstance(classes, int):
        classes = [classes]

    if max(classes) > 20 or min(classes) < 0:
        raise ValueError('Class indices must be between 0 and 20')

    r_classes = list(reversed(classes))
    if r_classes[-1] == 0:
        r_classes.insert(0, r_classes.pop(-1))

    # Support points
    np.random.seed(seed)
    if support is None:
        support = np.random.uniform(support_low, support_high, (1, n_nodes, 2))
    else:
        try:
            assert support.shape == (1, n_nodes, 2)
        except AssertionError:
            print('The given support doesn\'t have shape (1, n_nodes, 2) as'
                  'expected. Attempting to reshape.')
            support = support.reshape(1, n_nodes, 2)

    # Compute node features
    node_features = []
    # Other node features
    for idx, i in enumerate(r_classes):
        if i == 0:
            concept_0 = np.repeat(support, n_samples_in_class, 0)
            noise_0 = np.random.normal(0, 1, (n_samples_in_class, n_nodes, 2))
            class_0 = concept_0 + noise_0
            node_features.append(class_0)
        else:
            radius = 10. * ((2./3.) ** (drift_amount * (i - 1)))
            phase = np.random.uniform(0, 2 * np.pi, (n_nodes, 1))
            perturb_i_x = radius * np.cos(phase)
            perturb_i_y = radius * np.sin(phase)
            perturb_i = np.concatenate((perturb_i_x, perturb_i_y), axis=-1)
            support_i = support + perturb_i
            concept_i = np.repeat(support_i, n_samples_in_class, 0)
            noise_i = np.random.normal(0, 1, (n_samples_in_class, n_nodes, 2))
            class_i = concept_i + noise_i
            node_features.append(class_i)
    node_features = np.array(node_features).reshape(-1, n_nodes, 2)

    # Compute adjacency matrices
    adjacency = []
    for nf in node_features:
        tri = Delaunay(nf)
        edges_explicit = np.concatenate((tri.vertices[:, :2],
                                         tri.vertices[:, 1:],
                                         tri.vertices[:, ::2]), axis=0)
        adj = np.zeros((n_nodes, n_nodes))
        adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
        adj = np.clip(adj + adj.T, 0, 1)
        adjacency.append(adj)
    adjacency = np.array(adjacency)

    # Compute labels
    labels = np.repeat(classes, n_samples_in_class)
    if one_hot_labels:
        labels = label_to_one_hot(labels, labels=classes)

    if return_type is 'numpy':
        return adjacency, node_features, labels
    elif return_type is 'networkx':
        graphs = numpy_to_nx(adjacency, node_features=node_features, nf_name='coords')
        return graphs, labels
    else:
        raise NotImplementedError
