from spektral.datasets import tud
from spektral.utils import numpy_to_batch, numpy_to_disjoint


def test_utils_data():
    # Load ENZYMES because we use it also in datasets tests
    A_list, X_list, y = tud.load_data('ENZYMES', clean=True)

    # Test numpy to batch
    X_batch, A_batch = numpy_to_batch(X_list, A_list)
    assert X_batch.ndim == 3
    assert A_batch.ndim == 3
    assert X_batch.shape[0] == A_batch.shape[0]
    assert X_batch.shape[1] == A_batch.shape[1] == A_batch.shape[2]

    # Test numpy to disjoint
    X_disj, A_disj, I_disj = numpy_to_disjoint(X_list, A_list)
    assert X_disj.ndim == 2
    assert A_disj.ndim == 2
    assert X_disj.shape[0] == A_disj.shape[0] == A_disj.shape[1]
