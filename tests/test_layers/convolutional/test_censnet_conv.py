import enum
import numpy as np
from tensorflow.keras import Input, Model, layers
import networkx as nx
import pytest

from spektral.layers import CensNetConv
from core import batch_size, F, S


NODE_CHANNELS = 8
"""
Number of node output channels to use for testing.
"""
EDGE_CHANNELS = 10
"""
Number of edge output channels to use for testing.
"""


@enum.unique
class Modes(enum.IntEnum):
    """
    Represents the data modes to use for testing.
    """

    SINGLE = enum.auto()
    BATCH = enum.auto()
    MIXED = enum.auto()


@pytest.fixture()
def random_graph_descriptors():
    """
    Creates a random graph to use, and computes its various descriptors.
    :return: The normalized graph laplacian, the normalized line graph
        laplacian, and the incidence matrix.
    """
    graph = nx.dense_gnm_random_graph(11, 50, seed=1337)
    line_graph = nx.line_graph(graph)

    node_laplacian = np.array(nx.normalized_laplacian_matrix(graph).todense())
    edge_laplacian = np.array(nx.normalized_laplacian_matrix(line_graph).todense())
    incidence = np.array(nx.incidence_matrix(graph).todense())

    return node_laplacian, edge_laplacian, incidence


@pytest.mark.parametrize("mode", [Modes.SINGLE, Modes.BATCH, Modes.MIXED])
def test_smoke(random_graph_descriptors, mode):
    """
    Tests that we can create a model with the layer, and it processes
    input data without crashing.

    :param random_graph_descriptors: Descriptors for the graph to use when
        testing.
    :param mode: The data mode to use for this test.
    """
    # Arrange.
    node_laplacian, edge_laplacian, incidence = random_graph_descriptors

    # Create node and edge features.
    node_feature_shape = (node_laplacian.shape[0], F)
    edge_feature_shape = (edge_laplacian.shape[0], S)
    if mode != Modes.SINGLE:
        # Add batch dimensions.
        node_feature_shape = (batch_size,) + node_feature_shape
        edge_feature_shape = (batch_size,) + edge_feature_shape

    if mode == Modes.BATCH:
        node_laplacian = np.stack([node_laplacian] * batch_size)
        edge_laplacian = np.stack([edge_laplacian] * batch_size)
        incidence = np.stack([incidence] * batch_size)

    node_features = np.random.normal(size=node_feature_shape)
    edge_features = np.random.normal(size=edge_feature_shape)

    # Create the model.
    node_input = Input(shape=node_features.shape[1:])
    laplacian_input = Input(shape=node_laplacian.shape[1:])
    edge_laplacian_input = Input(shape=edge_laplacian.shape[1:])
    incidence_input = Input(shape=incidence.shape[1:])
    edge_input = Input(shape=edge_features.shape[1:])

    next_nodes, next_edges = CensNetConv(NODE_CHANNELS, EDGE_CHANNELS, activation="relu")(
        (
            node_input,
            (laplacian_input, edge_laplacian_input, incidence_input),
            edge_input,
        )
    )

    model = Model(
        inputs=(
            node_input,
            edge_input,
            laplacian_input,
            edge_laplacian_input,
            incidence_input,
        ),
        outputs=(next_nodes, next_edges),
    )

    # Act.
    # Run the model.
    got_next_nodes, got_next_edges = model(
        [node_features, edge_features, node_laplacian, edge_laplacian, incidence]
    )

    # Assert.
    # Make sure that the output shapes are correct.
    got_node_shape = got_next_nodes.numpy().shape
    got_edge_shape = got_next_edges.numpy().shape
    assert got_node_shape == node_features.shape[:-1] + (NODE_CHANNELS,)
    assert got_edge_shape == edge_features.shape[:-1] + (EDGE_CHANNELS,)
