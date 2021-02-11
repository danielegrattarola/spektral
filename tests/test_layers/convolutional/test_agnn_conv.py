from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.AGNNConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 7, "trainable": True},
    "dense": False,
    "sparse": True,
    "edges": False,
}


def test_layer():
    run_layer(config)
    config["kwargs"]["trainable"] = False
    run_layer(config)
