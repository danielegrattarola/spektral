from spektral.layers import GraphConv, ChebConv, EdgeConditionedConv, GraphAttention, GraphConvSkip, ARMAConv, APPNP, \
    GraphSageConv, GINConv
from keras import backend as K, Model, Input
import numpy as np
import tensorflow as tf

SINGLE, BATCH, MIXED = 1, 2, 3  # Single, batch, mixed
LAYER_K_, MODES_K_, KWARGS_K_ = 'layer', 'modes', 'kwargs'
TESTS = [
    {
        LAYER_K_: GraphConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu'}
    },
    {
        LAYER_K_: ChebConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu'}
    },
    {
        LAYER_K_: GraphSageConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu'}
    },
    {
        LAYER_K_: EdgeConditionedConv,
        MODES_K_: [BATCH],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'edges': True}
    },
    {
        LAYER_K_: GraphAttention,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu'}
    },
    {
        LAYER_K_: GraphConvSkip,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu'}
    },
    {
        LAYER_K_: ARMAConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'T': 2, 'K': 2, 'recurrent': True}
    },
    {
        LAYER_K_: APPNP,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_channels': 16, 'H': 2}
    },
    {
        LAYER_K_: GINConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'n_hidden_layers': 1}
    }
]

sess = K.get_session()
batch_size = 32
N = 11
F = 7
S = 3

A = np.ones((N, N))
X = np.random.normal(size=(N, F))
E = np.random.normal(size=(N, N, S))


def _test_single_mode(layer, **kwargs):
    A_in = Input(shape=(None,))
    X_in = Input(shape=(F,))

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in])
    model = Model([X_in, A_in], output)

    sess.run(tf.global_variables_initializer())
    output = sess.run(model.output, feed_dict={X_in: X, A_in: A})

    assert output.shape == (N, kwargs['channels'])


def _test_batch_mode(layer, **kwargs):
    A_batch = np.stack([A] * batch_size)
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    feed_dict = {X_in: X_batch, A_in: A_batch}

    if kwargs.get('edges'):
        kwargs.pop('edges')
        E_batch = np.stack([E] * batch_size)
        E_in = Input(shape=(N, N, S))
        inputs.append(E_in)
        feed_dict[E_in] = E_batch

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    sess.run(tf.global_variables_initializer())
    output = sess.run(model.output, feed_dict=feed_dict)

    assert output.shape == (batch_size, N, kwargs['channels'])


def _test_mixed_mode(layer, **kwargs):
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N,))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    feed_dict = {X_in: X_batch, A_in: A}

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    sess.run(tf.global_variables_initializer())
    output = sess.run(model.output, feed_dict=feed_dict)

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
            elif mode == BATCH:
                _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == MIXED:
                _test_mixed_mode(test[LAYER_K_], **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])
