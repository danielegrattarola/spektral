from core import MODES, run_layer

from spektral import layers

config = {
    "layer": layers.GINConvBatch,
    "modes": [MODES["BATCH"]],
    "kwargs": {"channels": 8, "activation": "relu", "mlp_hidden": [16]},
    "dense": True,
    "sparse": True,
    "edges": False,
}


def test_layer():
    run_layer(config)
    config["kwargs"]["epsilon"] = 0.0
    run_layer(config)
