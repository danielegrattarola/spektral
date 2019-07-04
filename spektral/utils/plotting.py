from __future__ import absolute_import

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation

from .conversion import numpy_to_nx
from mpl_toolkits.mplot3d import Axes3D

HORIZONTAL_ROTATION = 30
VERTICAL_ROTATION = 5


# GRAPHS #######################################################################
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
    layout = deserialize_nx_layout(layout, nf_name=nf_name)
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


# 3D PLOTS   ###################################################################
def spherical_plot(data, r, keep=False, **kwargs):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')
    u_s = np.linspace(0, 2 * np.pi, 100)
    v_s = np.linspace(0, np.pi, 100)
    x_s = r * np.outer(np.cos(u_s), np.sin(v_s))
    y_s = r * np.outer(np.sin(u_s), np.sin(v_s))
    z_s = r * np.outer(np.ones(np.size(u_s)), np.cos(v_s))
    if not keep:
        plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_ylim(-(r * 1.1), (r * 1.1))
    ax.set_xlim(-(r * 1.1), (r * 1.1))
    ax.set_zlim(-(r * 1.1), (r * 1.1))
    ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                      linewidth=0.5)
    ax.scatter(0, 0, 0, c='k')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)

    return ax


def hyperbolic_plot(data, r, **kwargs):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')
    x_s = np.arange(-10, 10, 0.1)
    y_s = np.arange(-10, 10, 0.1)
    x_s, y_s = np.meshgrid(x_s, y_s)
    z_s = np.sqrt((x_s ** 2 + y_s ** 2) + r ** 2)
    plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    ax.set_zlim(data[:, 2].min() - 1, data[:, 2].max() + 1)
    ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                      linewidth=0.5)
    ax.scatter(0, 0, 0, c='k')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)

    return ax


def euclidean_plot(data, **kwargs):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')
    plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    ax.set_zlim(data[:, 2].min() - 1, data[:, 2].max() + 1)
    ax.scatter(0, 0, 0, c='k')
    ax.view_init(15, 70)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)

    return ax


# ANIMATIONS ###################################################################
def histogram_animation(data, window_size=1, stride=1, frames=None,
                        filename=None, fps=30):
    def update_hist(frm):
        plt.cla()
        plt.xlim(data[:, ...].min(), data[:, ...].max())
        plt.hist(data[frm * stride: frm * stride + window_size, ...],
                 bins=100, color=plt.cm.winter(float(frm) / frames),
                 density=True)

    fig = plt.figure()
    plt.xlim(data[:, ...].min(), data[:, ...].max())
    plt.hist(data[:window_size, ...], bins=100, color=plt.cm.winter(0.),
             density=True)

    ani = animation.FuncAnimation(fig, update_hist, frames=frames)
    if filename is not None:
        ani.save(filename, writer='imagemagick', fps=fps)
    return ani


def spherical_animation(data, r, window_size=1, stride=1, frames=None,
                        filename=None, fps=30):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')
    u_s = np.linspace(0, 2 * np.pi, 100)
    v_s = np.linspace(0, np.pi, 100)
    x_s = r * np.outer(np.cos(u_s), np.sin(v_s))
    y_s = r * np.outer(np.sin(u_s), np.sin(v_s))
    z_s = r * np.outer(np.ones(np.size(u_s)), np.cos(v_s))

    # Animation of embeddings as change happens
    def update_hist(frm):
        progress = float(frm) / frames
        start = frm * stride
        stop = start + window_size
        plt.cla()
        ax = plt.gca()
        ax.set_ylim(-(r * 1.1), (r * 1.1))
        ax.set_xlim(-(r * 1.1), (r * 1.1))
        ax.set_zlim(-(r * 1.1), (r * 1.1))
        ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                          linewidth=0.5)
        ax.scatter(0, 0, 0, c='k')
        ax.view_init(15, 70)
        ax.scatter(data[start:stop, 0], data[start:stop, 1],
                   data[start:stop, 2],
                   c=plt.cm.winter(progress),
                   marker='.', alpha=0.9)
        ax.view_init(15 + VERTICAL_ROTATION * progress,
                     70 + HORIZONTAL_ROTATION * progress)

    fig = plt.figure()
    plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_ylim(-(r * 1.1), (r * 1.1))
    ax.set_xlim(-(r * 1.1), (r * 1.1))
    ax.set_zlim(-(r * 1.1), (r * 1.1))
    ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                      linewidth=0.5)
    ax.scatter(0, 0, 0, c='k')
    ax.view_init(15, 70)
    ax.scatter(data[:window_size, 0],
               data[:window_size, 1],
               data[:window_size, 2],
               c=plt.cm.winter(0.), marker='.', alpha=0.9)

    ani = animation.FuncAnimation(fig, update_hist, frames=frames)
    if filename is not None:
        ani.save(filename, writer='imagemagick', fps=fps)

    return ani


def hyperbolic_animation(data, r, window_size=1, stride=1, frames=None,
                         filename=None, fps=30):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')
    x_s = np.arange(-10, 10, 0.1)
    y_s = np.arange(-10, 10, 0.1)
    x_s, y_s = np.meshgrid(x_s, y_s)
    z_s = np.sqrt((x_s ** 2 + y_s ** 2) + r ** 2)

    # Animation of embeddings as change happens
    def update_hist(frm):
        progress = float(frm) / frames
        start = frm * stride
        stop = start + window_size
        plt.cla()
        ax = plt.gca()
        ax.set_xlim(data[:, 0].min() - 1,
                    data[:, 0].max() + 1)
        ax.set_ylim(data[:, 1].min() - 1,
                    data[:, 1].max() + 1)
        ax.set_zlim(data[:, 2].min() - 1,
                    data[:, 2].max() + 1)
        ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                          linewidth=0.5)
        ax.scatter(0, 0, 0, c='k')
        ax.view_init(15, 70)
        ax.scatter(data[start:stop, 0],
                   data[start:stop, 1],
                   data[start:stop, 2],
                   c=plt.cm.winter(progress),
                   marker='.', alpha=0.9)
        ax.view_init(15 + VERTICAL_ROTATION * progress,
                     70 + HORIZONTAL_ROTATION * progress)

    fig = plt.figure()
    plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_xlim(data[:, 0].min() - 1,
                data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1,
                data[:, 1].max() + 1)
    ax.set_zlim(data[:, 2].min() - 1,
                data[:, 2].max() + 1)
    ax.plot_wireframe(x_s, y_s, z_s, rstride=10, cstride=10, color='k',
                      linewidth=0.5)
    ax.scatter(0, 0, 0, c='k')
    ax.view_init(15, 70)
    ax.scatter(data[:window_size, 0],
               data[:window_size, 1],
               data[:window_size, 2],
               c=plt.cm.winter(0.), marker='.', alpha=0.9)

    ani = animation.FuncAnimation(fig, update_hist, frames=frames)
    if filename is not None:
        ani.save(filename, writer='imagemagick', fps=fps)

    return ani


def euclidean_animation(data, window_size=1, stride=1, frames=None,
                        filename=None, fps=30):
    if data.shape[-1] != 3:
        raise ValueError('data must have shape (n_samples, 3).')

    # Animation of embeddings as change happens
    def update_hist(frm):
        progress = float(frm) / frames
        start = frm * stride
        stop = start + window_size
        plt.cla()
        ax = plt.gca()
        ax.set_xlim(data[:, 0].min() - 1,
                    data[:, 0].max() + 1)
        ax.set_ylim(data[:, 1].min() - 1,
                    data[:, 1].max() + 1)
        ax.set_zlim(data[:, 2].min() - 1,
                    data[:, 2].max() + 1)
        ax.scatter(0, 0, 0, c='k')
        ax.view_init(15, 70)
        ax.scatter(data[start:stop, 0],
                   data[start:stop, 1],
                   data[start:stop, 2],
                   c=plt.cm.winter(progress),
                   marker='.', alpha=0.9)
        ax.view_init(15 + VERTICAL_ROTATION * progress,
                     70 + HORIZONTAL_ROTATION * progress)

    fig = plt.figure()
    plt.gcf().add_subplot(111, projection='3d')
    ax = plt.gca()
    ax.set_xlim(data[:, 0].min() - 1,
                data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1,
                data[:, 1].max() + 1)
    ax.set_zlim(data[:, 2].min() - 1,
                data[:, 2].max() + 1)
    ax.scatter(0, 0, 0, c='k')
    ax.view_init(15, 70)
    ax.scatter(data[:window_size, 0],
               data[:window_size, 1],
               data[:window_size, 2],
               c=plt.cm.winter(0.), marker='.', alpha=0.9)

    ani = animation.FuncAnimation(fig, update_hist, frames=frames)
    if filename is not None:
        ani.save(filename, writer='imagemagick', fps=fps)

    return ani


# Utils
def coordinates_layout_closure(nf_name):
    def coordinates_layout(nx_graph):
        return nx.get_node_attributes(nx_graph, nf_name)

    return coordinates_layout


def deserialize_nx_layout(layout, nf_name=None):
    if isinstance(layout, str):
        if layout in nx.layout.__all__:
            return eval('nx.{}'.format(layout))
        elif layout == 'coordinates':
            if nf_name is None:
                raise ValueError('nf_name cannot be None')
            return coordinates_layout_closure(nf_name)
        else:
            raise ValueError('layout must be in nx.layout.__all__ or \'coordinates\'')
