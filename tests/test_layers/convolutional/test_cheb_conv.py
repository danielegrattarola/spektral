from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.ChebConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {"K": 3, "channels": 8, "activation": "relu"},
    "dense": True,
    "sparse": True,
}


def test_layer():
    run_layer(config)
