import numpy as np
from core import MODES, run_layer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from spektral import layers
from spektral.layers import XENetConv, XENetDenseConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

# Not using these tests because they assume certain behaviors that we
# don't follow
"""
dense_config = {
    "layer": layers.XENetDenseConv,
    "modes": [MODES["BATCH"],],
    "kwargs": {"kernel_network": [8], "stack_channels": [2, 4], "node_channels": 64, "edge_channels": 16, "channels" : 64 },
    "dense": True,
    "sparse": True,
    "edges": True,
}

sparse_config = {
    "layer": layers.XENetConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"kernel_network": [8], "stack_channels": [2, 4], "node_channels": 64, "edge_channels": 16, "channels": 64 },
    "dense": True,
    "sparse": True,
    "edges": True,
}
"""


def test_sparse_model_sizes():
    """
    This is a sanity check to make sure we have the same number of operations that we intend to have
    """
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None,), name="A_in", sparse=True)
    E_in = Input(shape=(S,), name="E_in")

    x = np.ones(shape=(N, F))
    a = np.ones(shape=(N, N))
    a = sp_matrix_to_sp_tensor(a)
    e = np.ones(shape=(N * N, S))

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        print(model.count_params())
        assert model.count_params() == expected_size
        # for test coverage:
        model([x, a, e])

    X, E = XENetConv([5], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # int vs list: 5 vs [5]
    X, E = XENetConv(5, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # t = (4+4+3+3+1)*5 =  75    # Stack Conv
    # x = (4+5+5+1)*10  = 150    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 350

    X, E = XENetConv(5, 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 362)
    # t = (4+4+3+3+1)*5 =  75
    # a = (5+1)*1   *2  =  12    # Attention
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 362

    X, E = XENetConv([50, 5], 10, 20, True)([X_in, A_in, E_in])
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
    X_in = Input(shape=(N, F), name="X_in")
    A_in = Input(shape=(N, N), sparse=False, name="A_in")
    E_in = Input(shape=(N, N, S), name="E_in")

    x = np.ones(shape=(1, N, F))
    a = np.ones(shape=(1, N, N))
    e = np.ones(shape=(1, N, N, S))
    a[0][1][2] = 0
    a[0][2][1] = 0

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        print(model.count_params())
        assert model.count_params() == expected_size
        # for test coverage:
        model.predict([x, a, e])

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


if __name__ == "__main__":
    test_sparse_model_sizes()
    test_dense_model_sizes()
