import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from spektral.layers import Disjoint2Batch, SparseDropout, ops


def test_Disjoint2Batch():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 2]])
    I = np.array([0, 0, 0, 1, 1])
    A_data = [1, 1, 1, 1, 1]
    A_row = [0, 1, 2, 3, 4]
    A_col = [1, 0, 1, 4, 3]
    A = ops.sp_matrix_to_sp_tensor(
        sp.csr_matrix((A_data, (A_row, A_col)), shape=(5, 5))
    )

    expected_X = np.array(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]]
    )
    expected_A = np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    result_X, result_A = Disjoint2Batch()((X, A, I))
    assert np.allclose(result_A, expected_A)
    assert np.allclose(result_X, expected_X)


def test_sparse_dropout():
    """Ensure SparseDropout.sparse_dropout gradients don't throw issues."""
    n = 5
    rate = 0.5
    with tf.GradientTape() as tape:
        values = tf.Variable(tf.random.uniform((n,)))
        tape.watch(values)
        st = tf.sparse.eye(5)
        st = tf.SparseTensor(st.indices, values, st.dense_shape)
        st = SparseDropout(0.5)(st, training=True)
        loss = tf.sparse.reduce_sum(st)
    grad = tape.gradient(loss, values)
    assert grad is not None
