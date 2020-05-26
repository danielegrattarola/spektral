import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Input, Model

from spektral.layers import TopKPool, MinCutPool, DiffPool, SAGPool

tf.keras.backend.set_floatx('float64')

SINGLE, BATCH, DISJOINT = 1, 2, 3  # Single, batch, disjoint
LAYER_K_, MODES_K_, KWARGS_K_ = 'layer', 'modes', 'kwargs'
TESTS = [
    {
        LAYER_K_: TopKPool,
        MODES_K_: [SINGLE, DISJOINT],
        KWARGS_K_: {'ratio': 0.5, 'return_mask': True, 'sparse': True}
    },
    {
        LAYER_K_: SAGPool,
        MODES_K_: [SINGLE, DISJOINT],
        KWARGS_K_: {'ratio': 0.5, 'return_mask': True, 'sparse': True}
    },
    {
        LAYER_K_: MinCutPool,
        MODES_K_: [SINGLE, BATCH],
        KWARGS_K_: {'k': 5, 'return_mask': True, 'sparse': True}
    },
    {
        LAYER_K_: DiffPool,
        MODES_K_: [SINGLE, BATCH],
        KWARGS_K_: {'k': 5, 'return_mask': True, 'sparse': True}
    },

]

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


def _test_single_mode(layer, **kwargs):
    A = np.ones((N, N))
    X = np.random.normal(size=(N, F))
    sparse = kwargs.pop('sparse', None) is not None

    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in])
    model = Model([X_in, A_in], output)
    output = model([X, A])
    X_pool, A_pool, mask = output

    if 'ratio' in kwargs.keys():
        N_exp = kwargs['ratio'] * N
    elif 'k' in kwargs.keys():
        N_exp = kwargs['k']
    else:
        raise ValueError('Need k or ratio.')
    N_pool_expected = int(np.ceil(N_exp))
    N_pool_true = A_pool.shape[-1]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_batch_mode(layer, **kwargs):
    A = np.ones((batch_size, N, N))
    X = np.random.normal(size=(batch_size, N, F))

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in])
    model = Model([X_in, A_in], output)
    output = model([X, A])
    X_pool, A_pool, mask = output

    if 'ratio' in kwargs.keys():
        N_exp = kwargs['ratio'] * N
    elif 'k' in kwargs.keys():
        N_exp = kwargs['k']
    else:
        raise ValueError('Need k or ratio.')
    N_pool_expected = int(np.ceil(N_exp))
    N_pool_true = A_pool.shape[-1]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (batch_size, N_pool_expected, F)
    assert A_pool.shape == (batch_size, N_pool_expected, N_pool_expected)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_disjoint_mode(layer, **kwargs):
    A = sp.block_diag([np.ones((N1, N1)), np.ones(
        (N2, N2)), np.ones((N3, N3))]).todense()
    X = np.random.normal(size=(N, F))
    I = np.array([0] * N1 + [1] * N2 + [2] * N3).astype(int)
    sparse = kwargs.pop('sparse', None) is not None

    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    I_in = Input(shape=(), dtype=tf.int32)

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in, I_in])
    model = Model([X_in, A_in, I_in], output)
    output = model([X, A, I])
    X_pool, A_pool, I_pool, mask = output

    N_pool_expected = np.ceil(kwargs['ratio'] * N1) + \
                      np.ceil(kwargs['ratio'] * N2) + \
                      np.ceil(kwargs['ratio'] * N3)
    N_pool_expected = int(N_pool_expected)
    N_pool_true = A_pool.shape[0]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)
    assert I_pool.shape == (N_pool_expected,)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_get_config(layer, **kwargs):
    if kwargs.get('edges'):
        kwargs.pop('edges')
    layer_instance = layer(**kwargs)
    config = layer_instance.get_config()
    assert layer(**config)


def test_layers():
    for test in TESTS:
        for mode in test[MODES_K_]:
            if mode == SINGLE:
                _test_single_mode(test[LAYER_K_], **test[KWARGS_K_])
                if test[KWARGS_K_].pop('sparse', None):
                    _test_single_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == BATCH:
                _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == DISJOINT:
                _test_disjoint_mode(test[LAYER_K_], **test[KWARGS_K_])
                if test[KWARGS_K_].pop('sparse', None):
                    _test_disjoint_mode(test[LAYER_K_], **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])
