import enum

import networkx as nx
import numpy as np
import pytest
from core import A, F, S, batch_size
from tensorflow.keras import Input, Model

from spektral.layers import CensNetConv

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

    next_nodes, next_edges = CensNetConv(
        NODE_CHANNELS, EDGE_CHANNELS, activation="relu"
    )(
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


@pytest.mark.parametrize("mode", [Modes.SINGLE, Modes.BATCH, Modes.MIXED])
@pytest.mark.parametrize(("num_nodes", "num_edges"), [(5, 5), (4, 10), (1, 1)])
def test_output_shape(mode, num_nodes, num_edges):
    """
    Tests that we can compute the output shape correctly.

    :param mode: The data mode to use for this test.
    :param num_nodes: The number of nodes to use for the input.
    :param num_edges: The number of edges to use for the input.
    """
    # Arrange.
    # Create valid-looking input shapes.
    node_features_shape = (num_nodes, F)
    edge_features_shape = (num_edges, S)
    node_laplacian_shape = (num_nodes, num_nodes)
    edge_laplacian_shape = (num_edges, num_edges)
    incidence_shape = (num_nodes, num_edges)

    if mode != Modes.SINGLE:
        node_features_shape = (batch_size,) + node_features_shape
        edge_features_shape = (batch_size,) + edge_features_shape
    if mode == Modes.BATCH:
        node_laplacian_shape = (batch_size,) + node_laplacian_shape
        edge_laplacian_shape = (batch_size,) + edge_laplacian_shape
        incidence_shape = (batch_size,) + incidence_shape

    input_shape = (
        node_features_shape,
        (node_laplacian_shape, edge_laplacian_shape, incidence_shape),
        edge_features_shape,
    )

    # Create the layer.
    layer = CensNetConv(NODE_CHANNELS, EDGE_CHANNELS)

    # Act.
    got_output_shape = layer.compute_output_shape(input_shape)

    # Assert.
    # The output shape should be the same as the input, but with the correct
    # channel numbers.
    expected_node_feature_shape = node_features_shape[:-1] + (NODE_CHANNELS,)
    expected_edge_feature_shape = edge_features_shape[:-1] + (EDGE_CHANNELS,)
    assert got_output_shape == (
        expected_node_feature_shape,
        expected_edge_feature_shape,
    )


def test_get_config_round_trip():
    """
    Tests that it is possible to serialize a layer using `get_config()`,
    and then re-instantiate an identical one.
    """
    # Arrange.
    # Create the layer to test with.
    layer = CensNetConv(NODE_CHANNELS, EDGE_CHANNELS)

    # Act.
    config = layer.get_config()
    new_layer = CensNetConv(**config)

    # Assert.
    # The new layer should be the same.
    assert new_layer.node_channels == layer.node_channels
    assert new_layer.edge_channels == layer.edge_channels


def test_preprocess_smoke():
    """
    Tests that the preprocessing functionality does not crash.
    """
    # Act.
    node_laplacian, edge_laplacian, incidence = CensNetConv.preprocess(A)
