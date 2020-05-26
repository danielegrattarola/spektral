import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model

from spektral.layers import GlobalSumPool, GlobalAttnSumPool, \
    GlobalAttentionPool, GlobalAvgPool, GlobalMaxPool, SortPool

tf.keras.backend.set_floatx('float64')
batch_size = 32
N = 11
F = 7


def _check_output_and_model_output_shapes(true_shape, model_shape):
    assert len(true_shape) == len(model_shape)
    for i in range(len(true_shape)):
        if true_shape[i] == N:
            assert model_shape[i] in {N, None}
        elif true_shape[i] == batch_size:
            assert model_shape[i] in {batch_size, None}
        else:
            assert model_shape[i] == true_shape[i]


def _test_single_mode(layer, **kwargs):
    X = np.random.normal(size=(N, F))
    if 'target_shape' in kwargs:
        target_output_shape = kwargs.pop('target_shape')
    else:
        target_output_shape = (1, kwargs.get('channels', F))

    X_in = Input(shape=(F,))
    layer_instance = layer(**kwargs)
    output = layer_instance(X_in)
    model = Model(X_in, output)
    output = model(X)
    assert output.shape == target_output_shape
    assert output.shape == layer_instance.compute_output_shape(X.shape)
    _check_output_and_model_output_shapes(output.shape, model.output_shape)


def _test_batch_mode(layer, **kwargs):
    X = np.random.normal(size=(batch_size, N, F))
    if 'target_shape' in kwargs:
        target_output_shape = kwargs.pop('target_shape')
    else:
        target_output_shape = (batch_size, kwargs.get('channels', F))

    X_in = Input(shape=(N, F))
    layer_instance = layer(**kwargs)
    output = layer_instance(X_in)
    model = Model(X_in, output)
    output = model(X)
    assert output.shape == target_output_shape
    assert output.shape == layer_instance.compute_output_shape(X.shape)
    _check_output_and_model_output_shapes(output.shape, model.output_shape)


def _test_disjoint_mode(layer, **kwargs):
    X = np.random.normal(size=(batch_size * N, F))
    I = np.repeat(np.arange(batch_size), N).astype(np.int)
    if 'target_shape' in kwargs:
        target_output_shape = kwargs.pop('target_shape')
    else:
        target_output_shape = (batch_size, kwargs.get('channels', F))

    X_in = Input(shape=(F,))
    I_in = Input(shape=(), dtype=I.dtype)
    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, I_in])
    model = Model([X_in, I_in], output)
    output = model([X, I])
    assert output.shape == target_output_shape
    assert output.shape[1:] == layer_instance.compute_output_shape([X.shape, I.shape])[1:]
    _check_output_and_model_output_shapes(output.shape, model.output_shape)


def test_global_sum_pool():
    _test_single_mode(GlobalSumPool)
    _test_batch_mode(GlobalSumPool)
    _test_disjoint_mode(GlobalSumPool)


def test_global_avg_pool():
    _test_single_mode(GlobalAvgPool)
    _test_batch_mode(GlobalAvgPool)
    _test_disjoint_mode(GlobalAvgPool)


def test_global_max_pool():
    _test_single_mode(GlobalMaxPool)
    _test_batch_mode(GlobalMaxPool)
    _test_disjoint_mode(GlobalMaxPool)


def test_global_node_attention_pool():
    _test_single_mode(GlobalAttnSumPool)
    _test_batch_mode(GlobalAttnSumPool)
    _test_disjoint_mode(GlobalAttnSumPool)


def test_global_attention_pool():
    F_ = 10
    assert F_ != F
    _test_single_mode(GlobalAttentionPool, channels=F_)
    _test_batch_mode(GlobalAttentionPool, channels=F_)
    _test_disjoint_mode(GlobalAttentionPool, channels=F_)


def test_sort_pool():
    k = 6
    _test_single_mode(SortPool, k=k, target_shape=(k, F))
    _test_batch_mode(SortPool, k=k, target_shape=(batch_size, k, F))
    _test_disjoint_mode(SortPool, k=k, target_shape=(batch_size, k, F))
