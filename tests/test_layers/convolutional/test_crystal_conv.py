from core import run_layer, MODES
from spektral import layers

config = {
    "layer": layers.CrystalConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 7},
    "dense": False,
    "sparse": True,
    "edges": True
}


def test_layer():
    run_layer(config)
