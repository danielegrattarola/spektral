from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.APPNPConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {"channels": 8, "activation": "relu", "mlp_hidden": [16]},
    "dense": True,
    "sparse": True,
    "edges": False,
}


def test_layer():
    run_layer(config)
