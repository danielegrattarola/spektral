from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.GTVConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"]],
    "kwargs": {
        "channels": 8,
        "delta_coeff": 1.0,
        "epsilon": 0.001,
        "activation": "relu",
    },
    "dense": True,
    "sparse": True,
    "edges": False,
}


def test_layer():
    run_layer(config)
