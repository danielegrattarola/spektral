import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Input, Model

from spektral.layers.ops import sp_matrix_to_sp_tensor
from tests.test_layers.convolutional.core import _test_get_config

tf.keras.backend.set_floatx("float64")

MODES = {
    'SINGLE': 0,
    'BATCH': 1,
    'DISJOINT': 2,
}

batch_size = 3
N1, N2, N3 = 4, 5, 2
N = N1 + N2 + N3
F = 7


def _check_output_and_model_output_shapes(true_shape, model_shape):
    assert len(true_shape) == len(model_shape)
    for i in range(len(true_shape)):
        assert len(true_shape[i]) == len(model_shape[i])
        for j in range(len(true_shape[i])):
            assert model_shape[i][j] in {true_shape[i][j], None}


def _check_number_of_nodes(N_pool_expected, N_pool_true):
    if N_pool_expected is not None:
        assert N_pool_expected == N_pool_true or N_pool_true is None


def _test_single_mode(layer, sparse=False, **kwargs):
    A = np.ones((N, N))
    X = np.random.normal(size=(N, F))

    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]

    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X, A]

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    X_pool, A_pool, mask = output
    if "ratio" in kwargs.keys():
        N_exp = kwargs["ratio"] * N
    elif "k" in kwargs.keys():
        N_exp = kwargs["k"]
    else:
        raise ValueError("Need k or ratio.")
    N_pool_expected = int(np.ceil(N_exp))
    N_pool_true = A_pool.shape[-1]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_batch_mode(layer, **kwargs):
    A_batch = np.ones((batch_size, N, N))
    X_batch = np.random.normal(size=(batch_size, N, F))

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A_batch]

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    X_pool, A_pool, mask = output
    if "ratio" in kwargs.keys():
        N_exp = kwargs["ratio"] * N
    elif "k" in kwargs.keys():
        N_exp = kwargs["k"]
    else:
        raise ValueError("Need k or ratio.")
    N_pool_expected = int(np.ceil(N_exp))
    N_pool_true = A_pool.shape[-1]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (batch_size, N_pool_expected, F)
    assert A_pool.shape == (batch_size, N_pool_expected, N_pool_expected)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_disjoint_mode(layer, sparse=False, **kwargs):
    A = sp.block_diag(
        [np.ones((N1, N1)), np.ones((N2, N2)), np.ones((N3, N3))]
    ).todense()
    X = np.random.normal(size=(N, F))
    I = np.array([0] * N1 + [1] * N2 + [2] * N3).astype(int)

    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    I_in = Input(shape=(), dtype=tf.int32)
    inputs = [X_in, A_in, I_in]

    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A), I]
    else:
        input_data = [X, A, I]

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    X_pool, A_pool, I_pool, mask = output
    N_pool_expected = (
        np.ceil(kwargs["ratio"] * N1)
        + np.ceil(kwargs["ratio"] * N2)
        + np.ceil(kwargs["ratio"] * N3)
    )
    N_pool_expected = int(N_pool_expected)
    N_pool_true = A_pool.shape[0]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)
    assert I_pool.shape == (N_pool_expected,)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def run_layer(config):
    for mode in config["modes"]:
        if mode == MODES["SINGLE"]:
            if config["dense"]:
                _test_single_mode(config["layer"], **config["kwargs"])
            if config["sparse"]:
                _test_single_mode(config["layer"], sparse=True, **config["kwargs"])
        elif mode == MODES["BATCH"]:
            _test_batch_mode(config["layer"], **config["kwargs"])
        elif mode == MODES["DISJOINT"]:
            if config["dense"]:
                _test_disjoint_mode(config["layer"], **config["kwargs"])
            if config["sparse"]:
                _test_disjoint_mode(config["layer"], sparse=True, **config["kwargs"])
    _test_get_config(config["layer"], **config["kwargs"])