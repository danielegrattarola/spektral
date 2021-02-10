from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.GraphSageConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 8, "activation": "relu"},
    "dense": False,
    "sparse": True,
}


def test_layer():
    run_layer(config)
