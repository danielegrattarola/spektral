import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model

from spektral import layers
from spektral.layers.ops import sp_matrix_to_sp_tensor

tf.keras.backend.set_floatx("float64")
SINGLE, BATCH, MIXED = 1, 2, 3  # Single, batch, mixed
LAYER_K_, MODES_K_, KWARGS_K_ = "layer", "modes", "kwargs"
batch_size = 32
N = 11
F = 7
S = 3
A = np.ones((N, N))
X = np.random.normal(size=(N, F))
E = np.random.normal(size=(N, N, S))
E_single = np.random.normal(size=(N * N, S))


"""
Each entry in TESTS represent a test to be run for a particular Layer.
Each config dictionary has the form: 
{
    LAYER_K_: LayerClass,
    MODES_K_: [...],
    KWARGS_K_: {...},
},

LAYER_K_ is the class of the layer to be tested.
 
MODES_K_ is a list containing the data modes supported by the model, and should 
be at least one of: SINGLE, MIXED, BATCH. 

KWARGS_K_ is a dictionary containing: 
    - all keywords to be passed to the layer (including mandatory ones);
    - an optional entry 'edges': True if the layer supports edge attributes; 
    - an optional entry 'sparse': [...], indicating whether the layer supports 
    sparse or dense inputs as a bool (e.g., 'sparse': [False, True] will 
    test the layer on both dense and sparse adjacency matrix; 'sparse': [True] 
    will only test for sparse). By default, each layer is tested only on dense
    inputs. Batch mode only tests for dense inputs. 

The testing loop will create a simple 1-layer model and run it in single, mixed, 
and batch mode according the what specified in MODES_K_ in the testing config. 
The loop will check: 
    - that the model does not crash; 
    - that the output shape is pre-computed correctly; 
    - that the real output shape is correct; 
    - that the get_config() method works correctly (i.e., it is possible to 
    re-instatiate a layer using LayerClass(**layer_instance.get_config())).
"""

TESTS = [
    {
        LAYER_K_: layers.GCNConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.ChebConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "K": 3,
            "channels": 8,
            "activation": "relu",
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.GraphSageConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": 8, "activation": "relu", "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.ECCConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "kernel_network": [8],
            "channels": 8,
            "activation": "relu",
            "edges": True,
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.GATConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "attn_heads": 2,
            "concat_heads": False,
            "activation": "relu",
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.GCSConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.ARMAConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "order": 2,
            "iterations": 2,
            "share_weights": True,
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.APPNPConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "mlp_hidden": [16],
            "sparse_support": [False, True],
        },
    },
    {
        LAYER_K_: layers.GINConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "mlp_hidden": [16],
            "sparse_support": [True],
        },
    },
    {
        LAYER_K_: layers.DiffusionConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "tanh",
            "num_diffusion_steps": 5,
            "sparse_support": [False],
        },
    },
    {
        LAYER_K_: layers.GatedGraphConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": 10, "n_layers": 3, "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.AGNNConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": F, "trainable": True, "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.TAGConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": F, "K": 3, "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.CrystalConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": F, "edges": True, "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.EdgeConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {
            "channels": 8,
            "activation": "relu",
            "mlp_hidden": [16],
            "sparse_support": [True],
        },
    },
    {
        LAYER_K_: layers.GeneralConv,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": 256, "sparse_support": [True]},
    },
    {
        LAYER_K_: layers.MessagePassing,
        MODES_K_: [SINGLE, MIXED],
        KWARGS_K_: {"channels": F, "sparse_support": [True]},
    },
]


def _test_single_mode(layer, **kwargs):
    sparse = kwargs.pop("sparse", False)
    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X, A]

    if kwargs.pop("edges", None):
        E_in = Input(shape=(S,))
        inputs.append(E_in)
        input_data.append(E_single)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, kwargs["channels"])


def _test_batch_mode(layer, **kwargs):
    A_batch = np.stack([A] * batch_size)
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A_batch]

    if kwargs.pop("edges", None):
        E_batch = np.stack([E] * batch_size)
        E_in = Input(shape=(N, N, S))
        inputs.append(E_in)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs["channels"])


def _test_mixed_mode(layer, **kwargs):
    sparse = kwargs.pop("sparse", False)
    X_batch = np.stack([X] * batch_size)
    A_in = Input(shape=(N,), sparse=sparse)
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X_batch, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X_batch, A]

    if kwargs.pop("edges", None):
        E_in = Input(
            shape=(
                N * N,
                S,
            )
        )
        inputs.append(E_in)
        E_batch = np.stack([E_single] * batch_size)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs["channels"])


def _test_get_config(layer, **kwargs):
    if kwargs.get("edges"):
        kwargs.pop("edges")
    layer_instance = layer(**kwargs)
    config = layer_instance.get_config()
    layer_instance_new = layer(**config)
    config_new = layer_instance_new.get_config()
    config.pop("name")
    config_new.pop("name")

    # Remove 'name' if we have advanced activations (needed for GeneralConv)
    if "activation" in config and "class_name" in config["activation"]:
        config["activation"]["config"].pop("name")
        config_new["activation"]["config"].pop("name")

    assert config_new == config


def test_layers():
    for test in TESTS:
        sparse = test[KWARGS_K_]["sparse_support"]
        for mode in test[MODES_K_]:
            if mode == SINGLE:
                for s in sparse:
                    _test_single_mode(test[LAYER_K_], sparse=s, **test[KWARGS_K_])
            elif mode == BATCH:
                _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == MIXED:
                for s in sparse:
                    _test_mixed_mode(test[LAYER_K_], sparse=s, **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])
