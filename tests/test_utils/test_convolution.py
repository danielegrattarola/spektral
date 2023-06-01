import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from spektral.utils import convolution

g = nx.generators.erdos_renyi_graph(10, 0.2)
adj_sp = nx.adjacency_matrix(g).astype("f")
adj = adj_sp.A.astype("f")
degree = np.diag([d[1] for d in nx.degree(g)])
tol = 1e-6


def _run_dense_op(op, *args, **kwargs):
    for a in args:
        assert isinstance(a, np.ndarray)
    result = op(*args, **kwargs)
    if not isinstance(result, list):
        result = [result]
    for r in result:
        assert isinstance(r, np.ndarray)

    return result


def _run_sparse_op(op, *args, **kwargs):
    for a in args:
        assert isinstance(a, np.ndarray) or sp.issparse(a)
    result = op(*[sp.csr_matrix(a) for a in args], **kwargs)
    if not isinstance(result, list):
        result = [result]
    for r in result:
        assert sp.issparse(r)

    return result


def _check_results_equal(results_1, results_2):
    assert len(results_1) == len(results_2)
    for r_1, r_2 in zip(results_1, results_2):
        if sp.issparse(r_1):
            r_1 = r_1.A
        if sp.issparse(r_2):
            r_2 = r_2.A
        assert np.allclose(r_1, r_2, atol=tol)


def _check_op(op, *args, **kwargs):
    r_dense = _run_dense_op(op, *args, **kwargs)
    r_sparse = _run_sparse_op(op, *args, **kwargs)
    _check_results_equal(r_dense, r_sparse)


def test_degree():
    r_dense = _run_dense_op(convolution.degree_matrix, adj)
    r_sparse = _run_sparse_op(convolution.degree_matrix, adj_sp)
    _check_results_equal(r_dense, r_sparse)

    assert np.allclose(r_dense[0], degree)


def test_degree_power():
    _check_op(convolution.degree_power, adj, k=3)
    _check_op(convolution.degree_power, adj, k=-1.0)
    _check_op(convolution.degree_power, adj, k=-0.5)


def test_normalized_adjacency():
    _check_op(convolution.normalized_adjacency, adj, symmetric=False)
    _check_op(convolution.normalized_adjacency, adj, symmetric=True)


def test_laplacian():
    _check_op(convolution.laplacian, adj)


def test_normalized_laplacian():
    _check_op(convolution.normalized_laplacian, adj, symmetric=False)
    _check_op(convolution.normalized_laplacian, adj, symmetric=True)


def test_rescale_laplacian():
    l = convolution.laplacian(adj)
    _check_op(convolution.rescale_laplacian, l)
    _check_op(convolution.rescale_laplacian, l, lmax=2)


def test_gcn_filter():
    _check_op(convolution.gcn_filter, adj, symmetric=False)
    _check_op(convolution.gcn_filter, adj, symmetric=True)

    # Test batch mode
    _run_dense_op(convolution.gcn_filter, np.array([adj] * 3), symmetric=False)


@pytest.mark.parametrize(
    ("edges", "expected_t"),
    [
        ([(0, 1), (1, 2), (2, 3)], [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]]),
        (
            [(0, 1), (1, 2), (1, 3), (3, 0)],
            [[1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 1]],
        ),
        ([(0, 0), (1, 1)], [[1, 0], [0, 1]]),
    ],
    ids=["simple", "cycle", "self_loop"],
)
def test_incidence_matrix(edges, expected_t):
    """
    Tests that `incidence_matrix` produces correct results.
    :param edges: The graph to use for testing, as a list of edges.
    :param expected_t: The expected transformation matrix.
    """
    # Arrange.
    # Create the adjacency matrix.
    graph = nx.Graph(edges)
    adjacency = nx.to_numpy_array(graph)

    # Act.
    got_t = convolution.incidence_matrix(adjacency)

    # Assert.
    got_t = got_t.numpy()
    expected_t = np.array(expected_t)
    np.testing.assert_array_equal(expected_t, got_t)


def test_incidence_matrix_batch():
    """
    Tests that `incidence_matrix` works with multiple
    graphs in a batch.
    """
    # Arrange.
    # Create test graphs.
    graph1 = nx.to_numpy_array(nx.house_graph())
    graph2 = nx.to_numpy_array(nx.house_x_graph())

    # Act.
    # Get the incidence matrix for both graphs individually, and both together
    # as a batch.
    graph1_incidence = convolution.incidence_matrix(graph1)
    # The first graph will be padded with two extra rows columns so that it
    # can be batched with the other one, which has two more edges.
    graph1_incidence = np.pad(graph1_incidence, ((0, 0), (0, 2)))
    graph2_incidence = convolution.incidence_matrix(graph2)

    batch = np.stack([graph1, graph2], axis=0)
    batch_incidence = convolution.incidence_matrix(batch)

    # Assert.
    # Both ways of doing it should produce the same results.
    np.testing.assert_array_equal(graph1_incidence, batch_incidence[0])
    np.testing.assert_array_equal(graph2_incidence, batch_incidence[1])


def test_line_graph():
    """
    Tests that `line_graph` produces correct results.
    """
    # Arrange.
    # Compute the line graph using NX.
    graph = nx.house_graph()

    unordered_line_graph = nx.line_graph(graph)
    # NX puts the nodes in a weird order when making the line graph,
    # so we sort them to maintain consistency with the original graph and the
    # incidence matrix.
    line_graph = nx.Graph()
    line_graph.add_nodes_from(sorted(unordered_line_graph.nodes(data=True)))
    line_graph.add_edges_from(unordered_line_graph.edges(data=True))

    graph_incidence = nx.incidence_matrix(graph).toarray()
    true_line_graph_adj = nx.to_numpy_array(line_graph)

    # Act.
    got_line_graph_adj = convolution.line_graph(graph_incidence).numpy()

    # Assert
    np.testing.assert_array_equal(true_line_graph_adj, got_line_graph_adj)


def test_line_graph_batch():
    """
    Tests that `line_graph` works with multiple graphs in a batch.
    """
    # Arrange.
    # Create test graphs.
    graph1 = nx.house_graph()
    incidence1 = nx.incidence_matrix(graph1).toarray()
    # We will have to pad the first incidence matrix with two columns,
    # because the second graph has two extra edges.
    incidence1 = np.pad(incidence1, ((0, 0), (0, 2)))
    graph2 = nx.house_x_graph()
    incidence2 = nx.incidence_matrix(graph2).toarray()

    # Act.
    # Get the line graph for both graphs individually, and both together
    # as a batch.
    graph1_line = convolution.line_graph(incidence1)
    graph2_line = convolution.line_graph(incidence2)

    batch = np.stack([incidence1, incidence2], axis=0)
    line_graph_batch = convolution.line_graph(batch)

    # Assert.
    # Both ways of doing it should produce the same results.
    np.testing.assert_array_equal(graph1_line, line_graph_batch[0])
    np.testing.assert_array_equal(graph2_line, line_graph_batch[1])


def test_chebyshev_polynomial():
    _check_op(convolution.chebyshev_polynomial, adj, k=0)
    _check_op(convolution.chebyshev_polynomial, adj, k=1)
    _check_op(convolution.chebyshev_polynomial, adj, k=2)
    _check_op(convolution.chebyshev_polynomial, adj, k=5)


def test_chebyshev_filter():
    # Differences in computing the maximum eigenvalue lead to signifcantly different
    # values in the dense vs. sparse rescaled Laplacians.
    # We simply check that the ops run
    _run_dense_op(convolution.chebyshev_filter, adj, k=3, symmetric=False)
    _run_sparse_op(convolution.chebyshev_filter, adj, k=3, symmetric=True)


def test_add_self_loops():
    _check_op(convolution.add_self_loops, adj)
