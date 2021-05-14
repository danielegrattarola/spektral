from core import MODES, run_layer

from spektral import layers

from spektral.layers import XENetDenseConv, XENetSparseConv
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Not using these tests because they assume certain behaviors that we don't follow
'''
dense_config = {
    "layer": layers.XENetDenseConv,
    "modes": [MODES["BATCH"],],
    "kwargs": {"kernel_network": [8], "stack_channels": [2, 4], "node_channels": 64, "edge_channels": 16, "channels" : 64 },
    "dense": True,
    "sparse": True,
    "edges": True,
}

sparse_config = {
    "layer": layers.XENetSparseConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"kernel_network": [8], "stack_channels": [2, 4], "node_channels": 64, "edge_channels": 16, "channels": 64 },
    "dense": True,
    "sparse": True,
    "edges": True,
}
'''

def test_sparse_model_sizes():
    """
    This is a sanity check to make sure we have the same number of operations that we intend to have
    """
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None, ), name="A_in", sparse=True)
    E_in = Input(shape=(S, ), name="E_in")

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        print(model.count_params())
        assert model.count_params() == expected_size

    X, E = XENetSparseConv(1, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 126)
        
    X, E = XENetSparseConv([4], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 294)
        
    X, E = XENetSparseConv([5], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # int vs list: 5 vs [5]
    X, E = XENetSparseConv(5, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # t = (4+4+3+3+1)*5 =  75    # Stack Conv
    # x = (4+5+5+1)*10  = 150    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 350

    X, E = XENetSparseConv(5, 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 362)
    # t = (4+4+3+3+1)*5 =  75
    # a = (5+1)*1   *2  =  12    # Attention
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 362

    X, E = XENetSparseConv([50, 5], 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 1292)
    # t1 = (4+4+3+3+1)*50   =  750
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *2      =   12    # Attention
    # x = (4+5+5+1)*10      =  150
    # e = (5+1)*20          =  120
    # p                     =    5    # Prelu
    # total = t+x+e+p       = 1292

def test_dense_model_sizes():
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), sparse=False, name='A_in')
    E_in = Input(shape=(N, N, S), name='E_in')

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.count_params())
        model.summary()
        assert(model.count_params() == expected_size)

    X, E = XENetDenseConv(1, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 126)        
        
    X, E = XENetDenseConv([4], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 294)        
    
    X, E = XENetDenseConv([5], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # int vs list: 5 vs [5]
    X, E = XENetDenseConv(5, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # t = (4+4+3+3+1)*5 =  75    # Stack Conv
    # x = (4+5+5+1)*10  = 150    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 350

    X, E = XENetDenseConv(5, 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 362)
    # t = (4+4+3+3+1)*5 =  75
    # a = (5+1)*1   *2  =  12    # Attention
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 362

    X, E = XENetDenseConv([50, 5], 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 1292)
    # t1 = (4+4+3+3+1)*50   =  750
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *2      =   12    # Attention
    # x = (4+5+5+1)*10      =  150
    # e = (5+1)*20          =  120
    # p                     =    5    # Prelu
    # total = t+x+e+p       = 1292


def test_layer():
    #run_layer(dense_config)
    #run_layer(sparse_config)    
    test_sparse_model_sizes()
    test_dense_model_sizes()

if __name__ == '__main__':
    test_layer()
    
