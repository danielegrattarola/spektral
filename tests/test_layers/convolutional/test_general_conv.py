from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.GeneralConv,
    "modes": [MODES["SINGLE"], MODES["MIXED"]],
    "kwargs": {"channels": 256},
    "dense": False,
    "sparse": True,
}


def test_layer():
    run_layer(config)
    config["kwargs"]["activation"] = "relu"
    run_layer(config)
