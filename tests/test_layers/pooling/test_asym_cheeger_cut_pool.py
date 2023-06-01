from spektral import layers
from tests.test_layers.pooling.core import MODES, run_layer

config = {
    "layer": layers.AsymCheegerCutPool,
    "modes": [MODES["SINGLE"], MODES["BATCH"]],
    "kwargs": {
        "k": 5,
        "return_selection": True,
        "mlp_hidden": [32],
        "totvar_coeff": 1.0,
        "balance_coeff": 1.0,
    },
    "dense": True,
    "sparse": True,
}


def test_layer():
    run_layer(config)
