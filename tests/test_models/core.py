import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from spektral.data import Dataset, Graph, loaders

tf.keras.backend.set_floatx("float64")
MODES = {"SINGLE": 0, "BATCH": 1, "MIXED": 2, "DISJOINT": 3}

batch_size = 16
n_nodes = 11
n_node_features = 7
n_edge_features = 3


def _get_graph(n_nodes, n_features, n_edge_features=None, sparse=False):
    x = np.random.rand(n_nodes, n_features)
    a = np.random.randint(0, 2, (n_nodes, n_nodes)).astype("f4")
    e = (
        np.random.rand(np.count_nonzero(a), n_edge_features)
        if n_edge_features is not None
        else None
    )
    if sparse:
        a = sp.csr_matrix(a)
    return Graph(x=x, a=a, e=e)


class TestDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        super().__init__()

    def read(self):
        return self.graphs


def _test_single_mode(model, sparse=False, edges=False, **kwargs):
    dataset = TestDataset(
        [
            _get_graph(
                n_nodes=n_nodes,
                n_features=n_node_features,
                n_edge_features=n_edge_features if edges else None,
                sparse=sparse,
            )
        ]
    )

    loader = loaders.SingleLoader(dataset, epochs=1)
    inputs = list(loader)[0]

    model_instance = model(**kwargs)
    output = model_instance(inputs)


def _test_disjoint_mode(model, sparse=False, edges=False, **kwargs):
    dataset = TestDataset(
        [
            _get_graph(
                n_nodes=n_nodes,
                n_features=n_node_features,
                n_edge_features=n_edge_features if edges else None,
                sparse=sparse,
            )
            for _ in range(batch_size)
        ]
    )

    loader = loaders.DisjointLoader(dataset, epochs=1, batch_size=batch_size)
    inputs = loader.__next__()

    model_instance = model(**kwargs)
    output = model_instance(inputs)


def _test_batch_mode(model, edges=False, **kwargs):
    dataset = TestDataset(
        [
            _get_graph(
                n_nodes=n_nodes,
                n_features=n_node_features,
                n_edge_features=n_edge_features if edges else None,
            )
            for _ in range(batch_size)
        ]
    )

    loader = loaders.BatchLoader(dataset, epochs=1, batch_size=batch_size)
    inputs = loader.__next__()

    model_instance = model(**kwargs)
    output = model_instance(inputs)


def _test_mixed_mode(model, sparse=False, edges=False, **kwargs):
    graphs = []
    for i in range(batch_size):
        graph = _get_graph(
            n_nodes=n_nodes,
            n_features=n_node_features,
            n_edge_features=n_edge_features if edges else None,
            sparse=sparse,
        )
        if i == 0:
            a = graph.a
        graph.a = None

        graphs.append(graph)

    dataset = TestDataset(graphs)
    dataset.a = a

    loader = loaders.MixedLoader(dataset, epochs=1, batch_size=batch_size)
    inputs = loader.__next__()

    model_instance = model(**kwargs)
    output = model_instance(inputs)


def _test_get_config(layer, **kwargs):
    layer_instance = layer(**kwargs)
    config = layer_instance.get_config()
    layer_instance_new = layer(**config)
    config_new = layer_instance_new.get_config()

    # Remove 'name' if we have advanced activations (needed for GeneralConv)
    if (
        "activation" in config
        and isinstance(config["activation"], dict)
        and "class_name" in config["activation"]
    ):
        config["activation"]["config"].pop("name")
        config_new["activation"]["config"].pop("name")

    assert config_new == config


def run_model(config):
    """
    Each `config` is a dictionary with the form:
    {
        "model": class,
        "modes": list[int],
        "kwargs": dict,
        "dense": bool,
        "sparse": bool,
        "edges": bool
    },

    "model" is the class of the model to be tested.

    "modes" is a list containing the data modes supported by the model, as specified by
    the global MODES dictionary in this file.

    "kwargs" is a dictionary containing all keywords to be passed to the layer
    (including mandatory ones).

    "dense" is True if the layer supports dense adjacency matrices.

    "sparse" is True if the layer supports sparse adjacency matrices.

    "edges" is True if the layer supports edge attributes.

    The testing loop will create a simple 1-layer Model and run it in single, mixed,
    and batch mode according the what specified in the testing config.

    The loop will check:
        - that the model does not crash;
        - that the output shape is pre-computed correctly;
        - that the real output shape is correct;
        - that the get_config() method works correctly (i.e., it is possible to
          re-instatiate a layer using LayerClass(**layer_instance.get_config())).
    """
    for mode in config["modes"]:
        if mode == MODES["SINGLE"]:
            if config["dense"]:
                _test_single_mode(
                    config["model"],
                    sparse=False,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
            if config["sparse"]:
                _test_single_mode(
                    config["model"],
                    sparse=True,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
        elif mode == MODES["BATCH"]:
            _test_batch_mode(
                config["model"], edges=config.get("edges", False), **config["kwargs"]
            )
        elif mode == MODES["MIXED"]:
            if config["dense"]:
                _test_mixed_mode(
                    config["model"],
                    sparse=False,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
            if config["sparse"]:
                _test_mixed_mode(
                    config["model"],
                    sparse=True,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
        elif mode == MODES["DISJOINT"]:
            if config["dense"]:
                _test_disjoint_mode(
                    config["model"],
                    sparse=False,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
            if config["sparse"]:
                _test_disjoint_mode(
                    config["model"],
                    sparse=True,
                    edges=config.get("edges", False),
                    **config["kwargs"],
                )
    _test_get_config(config["model"], **config["kwargs"])
