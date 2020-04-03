import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input

from spektral.layers import GraphConv, ChebConv, EdgeConditionedConv, GraphAttention, GraphConvSkip, ARMAConv, APPNP, \
    GraphSageConv, GINConv

tf.keras.backend.set_floatx('float64')
SINGLE, BATCH, MIXED = 1, 2, 3  # Single, batch, mixed
LAYER_K_, MODES_K_, KWARGS_K_ = 'layer', 'modes', 'kwargs'

TESTS = [
    {
        LAYER_K_: GraphConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': True},
    },
    {
        LAYER_K_: ChebConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'K': 3, 'channels': 8, 'activation': 'relu', 'sparse': True}
    },
    {
        LAYER_K_: GraphSageConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': True}
    },
    {
        LAYER_K_: EdgeConditionedConv,
        MODES_K_: [SINGLE, BATCH],
        KWARGS_K_: {'kernel_network': [8], 'channels': 8, 'activation': 'relu', 'edges': True, 'sparse': True}
    },
    {
        LAYER_K_: GraphAttention,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'attn_heads': 2, 'concat_heads': False, 'activation': 'relu', 'sparse': True}
    },
    {
        LAYER_K_: GraphConvSkip,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': True}
    },
    {
        LAYER_K_: ARMAConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'order': 2, 'iterations': 2, 'share_weights': True, 'sparse': True}
    },
    {
        LAYER_K_: APPNP,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_hidden': [16], 'sparse': True}
    },
    {
        LAYER_K_: GINConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_hidden': [16], 'sparse': True}
    }
]

batch_size = 32
N = 11
F = 7
S = 3

A = np.ones((N, N))
X = np.random.normal(size=(N, F))
E = np.random.normal(size=(N, N, S))


def _test_single_mode(layer, **kwargs):
    sparse = kwargs.pop('sparse', None) is not None
    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]
    input_data = [X, A]

    if kwargs.pop('edges', None):
        E_in = Input(shape=(None, S))
        inputs.append(E_in)
        input_data.append(E)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, kwargs['channels'])


def _test_batch_mode(layer, **kwargs):
    A_batch = np.stack([A] * batch_size)
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A_batch]

    if kwargs.pop('edges', None):
        E_batch = np.stack([E] * batch_size)
        E_in = Input(shape=(N, N, S))
        inputs.append(E_in)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs['channels'])


def _test_mixed_mode(layer, **kwargs):
    X_batch = np.stack([X] * batch_size)
    sparse = kwargs.pop('sparse', None) is not None
    A_in = Input(shape=(N,), sparse=sparse)
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A]

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs['channels'])


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
            elif mode == MIXED:
                _test_mixed_mode(test[LAYER_K_], **test[KWARGS_K_])
                if test[KWARGS_K_].pop('sparse', None):
                    _test_mixed_mode(test[LAYER_K_], **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])
