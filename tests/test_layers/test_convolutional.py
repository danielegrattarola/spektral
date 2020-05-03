import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input

from spektral.layers import GraphConv, ChebConv, EdgeConditionedConv, GraphAttention, \
    GraphConvSkip, ARMAConv, APPNP, GraphSageConv, GINConv, DiffusionConv, \
    GatedGraphConv, AGNNConv, TAGConv, CrystalConv, MessagePassing, EdgeConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

tf.keras.backend.set_floatx('float64')
SINGLE, BATCH, MIXED = 1, 2, 3  # Single, batch, mixed
LAYER_K_, MODES_K_, KWARGS_K_ = 'layer', 'modes', 'kwargs'
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
        LAYER_K_: GraphConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': [False, True]},
    },
    {
        LAYER_K_: ChebConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'K': 3, 'channels': 8, 'activation': 'relu', 'sparse': [False, True]}
    },
    {
        LAYER_K_: GraphSageConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': [False, True]}
    },
    {
        LAYER_K_: EdgeConditionedConv,
        MODES_K_: [SINGLE, BATCH],
        KWARGS_K_: {'kernel_network': [8], 'channels': 8, 'activation': 'relu',
                    'edges': True, 'sparse': [False, True]}
    },
    {
        LAYER_K_: GraphAttention,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'attn_heads': 2, 'concat_heads': False,
                    'activation': 'relu', 'sparse': [False, True]}
    },
    {
        LAYER_K_: GraphConvSkip,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'sparse': [False, True]}
    },
    {
        LAYER_K_: ARMAConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'order': 2, 'iterations': 2,
                    'share_weights': True, 'sparse': [False, True]}
    },
    {
        LAYER_K_: APPNP,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_hidden': [16],
                    'sparse': [False, True]}
    },
    {
        LAYER_K_: GINConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_hidden': [16],
                    'sparse': [True]}
    },
    {
        LAYER_K_: DiffusionConv,
        MODES_K_: [SINGLE, BATCH, MIXED],
        KWARGS_K_: {'channels': 8, 'activation': 'tanh', 'num_diffusion_steps': 5,
                    'sparse': [False]}
    },
    {
        LAYER_K_: GatedGraphConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 10, 'n_layers': 3, 'sparse': [True]}
    },
    {
        LAYER_K_: AGNNConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': F, 'trainable': True, 'sparse': [True]}
    },
    {
        LAYER_K_: TAGConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': F, 'K': 3, 'sparse': [True]}
    },
    {
        LAYER_K_: CrystalConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': F, 'edges': True, 'sparse': [True]}
    },
    {
        LAYER_K_: EdgeConv,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 8, 'activation': 'relu', 'mlp_hidden': [16],
                    'sparse': [True]}
    },
    {
        LAYER_K_: MessagePassing,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': F, 'sparse': [True]}
    },
]


def _test_single_mode(layer, **kwargs):
    sparse = kwargs.pop('sparse', False)
    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X, A]

    if kwargs.pop('edges', None):
        E_in = Input(shape=(S, ))
        inputs.append(E_in)
        input_data.append(E_single)

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
    sparse = kwargs.pop('sparse', False)
    X_batch = np.stack([X] * batch_size)
    A_in = Input(shape=(N,), sparse=sparse)
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X_batch, sp_matrix_to_sp_tensor(A)]
    else:
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
                if 'sparse' in test[KWARGS_K_]:
                    sparse = test[KWARGS_K_].pop('sparse')
                    for s in sparse:
                        _test_single_mode(test[LAYER_K_], sparse=s, **test[KWARGS_K_])
                else:
                    _test_single_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == BATCH:
                _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == MIXED:
                if 'sparse' in test[KWARGS_K_]:
                    sparse = test[KWARGS_K_].pop('sparse')
                    for s in sparse:
                        _test_mixed_mode(test[LAYER_K_], sparse=s, **test[KWARGS_K_])
                else:
                    _test_mixed_mode(test[LAYER_K_], **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])
