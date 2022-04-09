from spektral import layers
from tests.test_layers.pooling.core import MODES, run_layer

config = {
    "layer": layers.LaPool,
    "modes": [MODES["SINGLE"], MODES["DISJOINT"]],
    "kwargs": {"return_selection": True},
    "dense": False,
    "sparse": True,
}


def test_layer():
    run_layer(config)
