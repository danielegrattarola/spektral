from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.TAGConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 7, "K": 3},
    "dense": False,
    "sparse": True,
}


def test_layer():
    run_layer(config)
