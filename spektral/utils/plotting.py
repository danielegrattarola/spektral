import networkx as nx

from spektral.utils.conversion import numpy_to_nx


def plot_nx(nx_graph, nf_name=None, ef_name=None, layout='spring_layout',
            labels=True, **kwargs):
    """
    Plot a Networkx graph.
    :param nx_graph: a Networkx graph;
    :param nf_name: string, name of the node features to plot;
    :param ef_name: string, name of the edge features to plot;
    :param layout: string, type of layout for networkx (see `nx.layout.__all__`);
    :param labels: bool, plot node and edge labels;
    :param kwargs: extra arguments for nx.draw;
    :return: None
    """
    layout = _deserialize_nx_layout(layout, nf_name=nf_name)
    pos = layout(nx_graph)
    nx.draw(nx_graph, pos, **kwargs)
    if nf_name is not None:
        node_labels = nx.get_node_attributes(nx_graph, nf_name)
        if labels:
            nx.draw_networkx_labels(nx_graph, pos, labels=node_labels)
    if ef_name is not None:
        edge_labels = nx.get_edge_attributes(nx_graph, ef_name)
        if labels:
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)


def plot_numpy(A, X=None, E=None, nf_name=None, ef_name=None,
               layout='spring_layout', labels=True, **kwargs):
    """
    Plots a graph in matrix format (adjacency matrix, node features matrix, and
    edge features matrix).
    :param A: np.array, adjacency matrix of the graph;
    :param X: np.array, node features matrix of the graph;
    :param E: np.array, edge features matrix of the graph;
    :param nf_name: string, name of the node features to plot;
    :param ef_name: string, name of the edge features to plot;
    :param layout: string, type of layout for networkx (see `nx.layout.__all__`);
    :param labels: bool, plot node and edge labels;
    :param kwargs: extra arguments for nx.draw;
    :return: None
    """
    if X is not None and nf_name is None:
        nf_name = 'nf'
    if E is not None and ef_name is None:
        ef_name = 'ef'
    g = numpy_to_nx(A, X, E, nf_name, ef_name)
    plot_nx(g, nf_name, ef_name, layout=layout, labels=labels, **kwargs)


def _deserialize_nx_layout(layout, nf_name=None):
    if isinstance(layout, str):
        if layout in nx.layout.__all__:
            return eval('nx.{}'.format(layout))
        elif layout == 'coordinates':
            if nf_name is None:
                raise ValueError('nf_name cannot be None')
            return lambda nx_graph: nx.get_node_attributes(nx_graph, nf_name)
        else:
            raise ValueError('layout must be in nx.layout.__all__ or \'coordinates\'')
