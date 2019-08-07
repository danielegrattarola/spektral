import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras import backend as K

from spektral.layers import TopKPool, MinCutPool
import scipy.sparse as sp

sess = K.get_session()
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

    A_in = Input(shape=(None, ))
    X_in = Input(shape=(F,))

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in])
    model = Model([X_in, A_in], output)
    sess.run(tf.global_variables_initializer())
    output = sess.run(model.output, feed_dict={X_in: X, A_in: A})
    X_pool, A_pool, mask = output

    if 'ratio' in kwargs.keys():
        N_exp = kwargs['ratio'] * N
    elif 'k' in kwargs.keys():
        N_exp = kwargs['k']
    else:
        raise ValueError('Need k or ratio.')
    N_pool_expected = np.ceil(N_exp)
    N_pool_true = A_pool.shape[-1]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def _test_graph_mode(layer, **kwargs):
    A = sp.block_diag([np.ones((N1, N1)), np.ones((N2, N2)), np.ones((N3, N3))]).todense()
    X = np.random.normal(size=(N, F))
    I = np.array([0] * N1 + [1] * N2 + [2] * N3).astype(int)

    A_in = Input(shape=(None, ))
    X_in = Input(shape=(F,))
    I_in = Input(shape=(), dtype=tf.int32)

    layer_instance = layer(**kwargs)
    output = layer_instance([X_in, A_in, I_in])
    model = Model([X_in, A_in, I_in], output)
    sess.run(tf.global_variables_initializer())
    output = sess.run(model.output, feed_dict={X_in: X, A_in: A, I_in: I})
    X_pool, A_pool, I_pool, mask = output

    N_pool_expected = np.ceil(kwargs['ratio'] * N1) + \
                      np.ceil(kwargs['ratio'] * N2) + \
                      np.ceil(kwargs['ratio'] * N3)
    N_pool_true = A_pool.shape[0]

    _check_number_of_nodes(N_pool_expected, N_pool_true)

    assert X_pool.shape == (N_pool_expected, F)
    assert A_pool.shape == (N_pool_expected, N_pool_expected)
    assert I_pool.shape == (N_pool_expected, )

    output_shape = [o.shape for o in output]
    _check_output_and_model_output_shapes(output_shape, model.output_shape)


def test_top_k_pool():
    _test_single_mode(TopKPool, ratio=0.5, return_mask=True)
    _test_graph_mode(TopKPool, ratio=0.5, return_mask=True)

def test_mincut_pool():
    _test_single_mode(MinCutPool, k=5, return_mask=True)


