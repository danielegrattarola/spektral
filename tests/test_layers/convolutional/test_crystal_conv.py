from core import MODES, F, run_layer

from spektral import layers

config = {
    "layer": layers.CrystalConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": F},  # Set channels same as node features
    "dense": False,
    "sparse": True,
    "edges": True,
}


def test_layer():
    run_layer(config)
