from spektral import layers
from tests.test_layers.pooling.core import MODES, run_layer

config = {
    "layer": layers.TopKPool,
    "modes": [MODES["SINGLE"], MODES["DISJOINT"]],
    "kwargs": {"ratio": 0.5, "return_selection": True},
    "dense": False,
    "sparse": True,
}


def test_layer():
    run_layer(config)
