from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.ECCConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {"kernel_network": [8], "channels": 8, "activation": "relu"},
    "dense": True,
    "sparse": True,
    "edges": True,
}


def test_layer():
    run_layer(config)
