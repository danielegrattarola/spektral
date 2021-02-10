import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model

from spektral.layers.ops import sp_matrix_to_sp_tensor

tf.keras.backend.set_floatx("float64")
MODES = {
    "SINGLE": 0,
    "BATCH": 1,
    "MIXED": 2,
}

batch_size = 32
N = 11
F = 7
S = 3
A = np.ones((N, N))
X = np.random.normal(size=(N, F))
E = np.random.normal(size=(N, N, S))
E_single = np.random.normal(size=(N * N, S))


def _test_single_mode(layer, sparse=False, edges=False, **kwargs):
    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]

    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X, A]

    if edges:
        E_in = Input(shape=(S,))
        inputs.append(E_in)
        input_data.append(E_single)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, kwargs["channels"])


def _test_batch_mode(layer, edges=False, **kwargs):
    A_batch = np.stack([A] * batch_size)
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A_batch]

    if edges:
        E_batch = np.stack([E] * batch_size)
        E_in = Input(shape=(N, N, S))
        inputs.append(E_in)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs["channels"])


def _test_mixed_mode(layer, sparse=False, edges=False, **kwargs):
    X_batch = np.stack([X] * batch_size)
    A_in = Input(shape=(N,), sparse=sparse)
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]

    if sparse:
        input_data = [X_batch, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X_batch, A]

    if edges:
        E_in = Input(shape=(N * N, S))
        inputs.append(E_in)
        E_batch = np.stack([E_single] * batch_size)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs["channels"])


def _test_get_config(layer, **kwargs):
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

def _test_preprocess(layer):
    a_out = layer.preprocess(A)
    assert a_out.shape == A.shape



def run_layer(config):
    """
    Each `config` is a dictionary with the form:
    {
        "layer": class,
        "modes": list[int],
        "kwargs": dict,
        "dense": bool,
        "sparse": bool,
        "edges": bool
    },

    "layer" is the class of the layer to be tested.

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
                    config["layer"],
                    sparse=False,
                    edges=config.get("edges", False),
                    **config["kwargs"]
                )
            if config["sparse"]:
                _test_single_mode(
                    config["layer"],
                    sparse=True,
                    edges=config.get("edges", False),
                    **config["kwargs"]
                )
        elif mode == MODES["BATCH"]:
            _test_batch_mode(
                config["layer"], edges=config.get("edges", False), **config["kwargs"]
            )
        elif mode == MODES["MIXED"]:
            if config["dense"]:
                _test_mixed_mode(
                    config["layer"],
                    sparse=False,
                    edges=config.get("edges", False),
                    **config["kwargs"]
                )
            if config["sparse"]:
                _test_mixed_mode(
                    config["layer"],
                    sparse=True,
                    edges=config.get("edges", False),
                    **config["kwargs"]
                )
    _test_get_config(config["layer"], **config["kwargs"])
    _test_preprocess(config["layer"])

