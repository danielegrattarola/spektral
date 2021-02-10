from spektral import models
from tests.test_models.core import MODES, run_model

config = {
    "model": models.GCN,
    "modes": [MODES["SINGLE"], MODES["DISJOINT"], MODES["MIXED"], MODES["BATCH"]],
    "kwargs": {"n_labels": 32},
    "edges": False,
    "dense": True,
    "sparse": True,
}


def test_model():
    run_model(config)
