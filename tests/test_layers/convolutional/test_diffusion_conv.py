from core import run_layer, MODES
from spektral import layers

config = {
    "layer": layers.DiffusionConv,
    "modes": [MODES["SINGLE"], MODES["BATCH"], MODES["MIXED"]],
    "kwargs": {"channels": 8, "activation": "tanh", "num_diffusion_steps": 5},
    "dense": True,
    "sparse": False,
}


def test_layer():
    run_layer(config)
