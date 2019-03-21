from __future__ import absolute_import

import networkx as nx
import numpy as np

from .misc import pad_jagged_array, add_eye_jagged, add_eye_batch, flatten_list


# Available conversions: Numpy <-> Networkx <-> SDF


def nx_to_adj(graphs):
    """
    Converts a list of nx.Graphs to a rank 3 np.array of adjacency matrices
    of shape `(num_graphs, num_nodes, num_nodes)`.
    :param graphs: a nx.Graph, or list of nx.Graphs.
    :return: a rank 3 np.array of adjacency matrices.
    """
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]
    return np.array([np.array(nx.attr_matrix(g)[0]) for g in graphs])


def nx_to_node_features(graphs, keys, post_processing=None):
    """
    Converts a list of nx.Graphs to a rank 3 np.array of node features matrices
    of shape `(num_graphs, num_nodes, num_features)`. Optionally applies a
    post-processing function to each individual attribute in the nx Graphs.
    :param graphs: a nx.Graph, or a list of nx.Graphs;
    :param keys: a list of keys with which to index node attributes in the nx
    Graphs.
    :param post_processing: a list of functions with which to post process each
    attribute associated to a key. `None` can be passed as post-processing 
    function to leave the attribute unchanged.
    :return: a rank 3 np.array of feature matrices
    """
    if post_processing is not None:
        if len(post_processing) != len(keys):
            raise ValueError('post_processing must contain an element for each key')
        for i in range(len(post_processing)):
            if post_processing[i] is None:
                post_processing[i] = lambda x: x

    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    output = []
    for g in graphs:
        node_features = []
        for v in g.nodes.values():
            f = [v[key] for key in keys]
            if post_processing is not None:
                f = [op(_) for op, _ in zip(post_processing, f)]
            f = flatten_list(f)
            node_features.append(f)
        output.append(np.array(node_features))

    return np.array(output)


def nx_to_edge_features(graphs, keys, post_processing=None):
    """
    Converts a list of nx.Graphs to a rank 4 np.array of edge features matrices
    of shape `(num_graphs, num_nodes, num_nodes, num_features)`.
    Optionally applies a post-processing function to each attribute in the nx
    graphs.
    :param graphs: a nx.Graph, or a list of nx.Graphs;
    :param keys: a list of keys with which to index edge attributes.
    :param post_processing: a list of functions with which to post process each
    attribute associated to a key. `None` can be passed as post-processing 
    function to leave the attribute unchanged.
    :return: a rank 3 np.array of feature matrices
    """
    if post_processing is not None:
        if len(post_processing) != len(keys):
            raise ValueError('post_processing must contain an element for each key')
        for i in range(len(post_processing)):
            if post_processing[i] is None:
                post_processing[i] = lambda x: x

    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    output = []
    for g in graphs:
        edge_features = []
        for key in keys:
            ef = np.array(nx.attr_matrix(g, edge_attr=key)[0])
            if ef.ndim == 2:
                ef = ef[..., None]  # Make it three dimensional to concatenate
            edge_features.append(ef)
        if post_processing is not None:
            edge_features = [op(_) for op, _ in zip(post_processing, edge_features)]
        if len(edge_features) > 1:
            edge_features = np.concatenate(edge_features, axis=-1)
        else:
            edge_features = np.array(edge_features[0])
        output.append(edge_features)

    return np.array(output)


def nx_to_numpy(graphs, auto_pad=True, self_loops=True, nf_keys=None,
                ef_keys=None, nf_postprocessing=None, ef_postprocessing=None):
    """
    Converts a list of nx.Graphs to numpy format (adjacency, node attributes,
    and edge attributes matrices).
    :param graphs: a nx.Graph, or list of nx.Graphs;
    :param auto_pad: whether to zero-pad all matrices to have graphs with the
    same dimension (set this to true if you don't want to deal with manual
    batching for different-size graphs.
    :param self_loops: whether to add self-loops to the graphs.
    :param nf_keys: a list of keys with which to index node attributes. If None,
    returns None as node attributes matrix.
    :param ef_keys: a list of keys with which to index edge attributes. If None,
    returns None as edge attributes matrix.
    :param nf_postprocessing: a list of functions with which to post process each
    node attribute associated to a key. `None` can be passed as post-processing
    function to leave the attribute unchanged.
    :param ef_postprocessing: a list of functions with which to post process each
    edge attribute associated to a key. `None` can be passed as post-processing
    function to leave the attribute unchanged.
    :return:
    - adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`
    - node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`
    - edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`
    """
    adj = nx_to_adj(graphs)
    if nf_keys is not None:
        nf = nx_to_node_features(graphs, nf_keys, post_processing=nf_postprocessing)
    else:
        nf = None
    if ef_keys is not None:
        ef = nx_to_edge_features(graphs, ef_keys, post_processing=ef_postprocessing)
    else:
        ef = None

    if self_loops:
        if adj.ndim == 1:  # Jagged array
            adj = add_eye_jagged(adj)
            adj = np.array([np.clip(a_, 0, 1) for a_ in adj])
        else:  # Rank 3 tensor
            adj = add_eye_batch(adj)
            adj = np.clip(adj, 0, 1)

    if auto_pad:
        # Pad all arrays to represent k-nodes graphs
        k = max([_.shape[-1] for _ in adj])
        adj = pad_jagged_array(adj, (k, k))
        if nf is not None:
            nf = pad_jagged_array(nf, (k, -1))
        if ef is not None:
            ef = pad_jagged_array(ef, (k, k, -1))

    return adj, nf, ef


def numpy_to_nx(adj, node_features=None, edge_features=None, nf_name=None,
                ef_name=None):
    """
    Converts graphs in numpy format to a list of nx.Graphs.
    :param adj: adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`.
    If there is only one sample, the first dimension can be dropped.
    :param node_features: optional node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`.
    If there is only one sample, the first dimension can be dropped.
    :param edge_features: optional edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`
    If there is only one sample, the first dimension can be dropped.
    :param nf_name: optional name to assign to node attributes in the nx.Graphs
    :param ef_name: optional name to assign to edge attributes in the nx.Graphs
    :return: a list of nx.Graphs (or a single nx.Graph is there is only one sample)
    """
    if adj.ndim == 2:
        adj = adj[None, ...]
        if node_features is not None:
            node_features = node_features[None, ...]
            if node_features.ndim != 3:
                raise ValueError('node_features must have shape (batch, N, F) '
                                 'or (N, F).')
        if edge_features is not None:
            edge_features = edge_features[None, ...]
            if edge_features.ndim != 4:
                raise ValueError('edge_features must have shape (batch, N, N, S) '
                                 'or (N, N, S).')

    output = []
    for i in range(adj.shape[0]):
        g = nx.from_numpy_array(adj[i])
        g.remove_nodes_from(list(nx.isolates(g)))

        if node_features is not None:
            node_attrs = {n: node_features[i, n] for n in g.nodes}
            nx.set_node_attributes(g, node_attrs, nf_name)
        if edge_features is not None:
            edge_attrs = {e: edge_features[i, e[0], e[1]] for e in g.edges}
            nx.set_edge_attributes(g, edge_attrs, ef_name)
        output.append(g)

    if len(output) == 1:
        return output[0]
    else:
        return output
