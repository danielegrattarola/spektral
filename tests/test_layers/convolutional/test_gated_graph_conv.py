from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.GatedGraphConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 10, "n_layers": 3},
    "dense": False,
    "sparse": True,
    "edges": False,
}


def test_layer():
    run_layer(config)
