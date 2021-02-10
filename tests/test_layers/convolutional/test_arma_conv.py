from core import run_layer, MODES
from spektral import layers

config = {
    "layer": layers.ARMAConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {
        "channels": 8,
        "activation": "relu",
        "order": 2,
        "iterations": 2,
        "share_weights": True,
        "use_bias": True
    },
    "dense": True,
    "sparse": True,
}


def test_layer():
    run_layer(config)
    config["kwargs"]["use_bias"] = False
    run_layer(config)
