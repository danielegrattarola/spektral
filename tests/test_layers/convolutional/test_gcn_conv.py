from core import run_layer, MODES
from spektral import layers

config = {
    "layer": layers.GCNConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {"channels": 8, "activation": "relu"},
    "dense": True,
    "sparse": True,
}


def test_layer():
    run_layer(config)
