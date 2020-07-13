import numpy as np
from scipy.spatial import Delaunay

from spektral.utils import label_to_one_hot, numpy_to_nx

RETURN_TYPES = {'numpy', 'networkx'}


def generate_data(classes=0, n_samples_in_class=1000, n_nodes=7, support_low=0.,
                  support_high=10., drift_amount=1.0, one_hot_labels=True,
                  support=None, seed=None, return_type='numpy'):
    """
    Generates a dataset of Delaunay triangulations as described by
    [Zambon et al. (2017)](https://arxiv.org/abs/1706.06941).

    Node attributes are the 2D coordinates of the points.
    Two nodes are connected if they share an edge in the Delaunay triangulation.
    Labels represent the class of the graph (0 to 20, each class index i
    represent the "difficulty" of the classification problem 0 v. i. In other
    words, the higher the class index, the more similar the class is to class 0).

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
    :param return_type: `'numpy'` or `'networkx'`, data format to return;
    :return:
    - if `return_type='numpy'`, the adjacency matrix, node features, and
    an array containing labels;
    - if `return_type='networkx'`, a list of graphs in Networkx format, and an
    array containing labels;
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
    node_features = np.array(node_features).reshape((-1, n_nodes, 2))

    # Compute adjacency matrices
    adjacency = []
    for nf in node_features:
        adj = _compute_adj(nf)
        adjacency.append(adj)
    adjacency = np.array(adjacency)

    # Compute labels
    labels = np.repeat(classes, n_samples_in_class)
    if one_hot_labels:
        labels = label_to_one_hot(labels, labels=classes)

    if return_type == 'numpy':
        return adjacency, node_features, labels
    elif return_type == 'networkx':
        graphs = numpy_to_nx(adjacency, node_features=node_features, nf_name='coords')
        return graphs, labels
    else:
        raise NotImplementedError


def _compute_adj(x):
    """
    Computes the Delaunay triangulation of the given points
    :param x: array of shape (num_nodes, 2)
    :return: the computed adjacency matrix
    """
    tri = Delaunay(x)
    edges_explicit = np.concatenate((tri.vertices[:, :2],
                                     tri.vertices[:, 1:],
                                     tri.vertices[:, ::2]), axis=0)
    adj = np.zeros((x.shape[0], x.shape[0]))
    adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
    return np.clip(adj + adj.T, 0, 1)
