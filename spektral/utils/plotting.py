import networkx as nx

from spektral.utils.conversion import numpy_to_nx


def plot_nx(nx_graph, nf_name=None, ef_name=None, layout='spring_layout',
            labels=True, node_color='r', node_size=300, **kwds):
    """
    Plot the given Networkx graph.
    :param nx_graph: a Networkx graph
    :param nf_name: name of the relevant node feature to plot
    :param ef_name: name of the relevant edgee feature to plot
    :param layout: type of layout for networkx
    :param labels: plot labels
    :param node_color: color for the plotted nodes
    :param node_size: size of the plotted nodes
    :return: None
    """
    layout = _deserialize_nx_layout(layout, nf_name=nf_name)
    pos = layout(nx_graph)
    nx.draw(nx_graph, pos, node_color=node_color, node_size=node_size, **kwds)
    if nf_name is not None:
        node_labels = nx.get_node_attributes(nx_graph, nf_name)
        if labels:
            nx.draw_networkx_labels(nx_graph, pos, labels=node_labels)
    if ef_name is not None:
        edge_labels = nx.get_edge_attributes(nx_graph, ef_name)
        if labels:
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)


def plot_numpy(adj, node_features=None, edge_features=None, nf_name=None,
               ef_name=None, layout='spring_layout', labels=True,
               node_color='r', node_size=300, ):
    """
    Converts a graph in matrix format (i.e. with adjacency matrix, node features
    matrix, and edge features matrix) to the Networkx format, then plots it with
    plot_nx().
    :param adj: np.array, adjacency matrix of the graph 
    :param node_features: np.array, node features matrix of the graph
    :param edge_features: np.array, edge features matrix of the graph
    :param nf_name: name to assign to the node features
    :param ef_name: name to assign to the edge features
    :param layout: type of layout for networkx
    :param labels: plot labels
    :param node_color: color for the plotted nodes
    :param node_size: size of the plotted nodes
    :return: None
    """
    if node_features is not None and nf_name is None:
        nf_name = 'nf'
    if edge_features is not None and ef_name is None:
        ef_name = 'ef'
    g = numpy_to_nx(adj, node_features, edge_features, nf_name, ef_name)
    plot_nx(g, nf_name, ef_name, node_color=node_color, node_size=node_size, layout=layout, labels=labels)


# Utils
def _coordinates_layout_closure(nf_name):
    def coordinates_layout(nx_graph):
        return nx.get_node_attributes(nx_graph, nf_name)

    return coordinates_layout


def _deserialize_nx_layout(layout, nf_name=None):
    if isinstance(layout, str):
        if layout in nx.layout.__all__:
            return eval('nx.{}'.format(layout))
        elif layout == 'coordinates':
            if nf_name is None:
                raise ValueError('nf_name cannot be None')
            return _coordinates_layout_closure(nf_name)
        else:
            raise ValueError('layout must be in nx.layout.__all__ or \'coordinates\'')
