import numpy as np
import scipy.sparse as sp

from spektral.layers import ops, Disjoint2Batch


def test_Disjoint2Batch():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 2]])
    I = np.array([0, 0, 0, 1, 1])
    A_data = [1, 1, 1, 1, 1]
    A_row = [0, 1, 2, 3, 4]
    A_col = [1, 0, 1, 4, 3]
    A = ops.sp_matrix_to_sp_tensor(
        sp.coo_matrix((A_data, (A_row, A_col)), shape=(5, 5))
    )

    expected_X = np.array([[[1., 0.],
                            [0., 1.],
                            [1., 1.]],
                           [[0., 0.],
                            [1., 2.],
                            [0., 0.]]])
    expected_A = np.array([[[0., 1., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.]],
                           [[0., 1., 0.],
                            [1., 0., 0.],
                            [0., 0., 0.]]])

    result_X, result_A = Disjoint2Batch()((X, A, I))
    assert np.allclose(result_A, expected_A)
    assert np.allclose(result_X, expected_X)
