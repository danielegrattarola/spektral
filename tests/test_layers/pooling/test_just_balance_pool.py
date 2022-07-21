from spektral import layers
from tests.test_layers.pooling.core import MODES, run_layer

config = {
    "layer": layers.JustBalancePool,
    "modes": [MODES["SINGLE"], MODES["BATCH"]],
    "kwargs": {"k": 5, "return_selection": True},
    "dense": True,
    "sparse": True,
}


def test_layer():
    run_layer(config)
