import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from spektral.layers import ops
from spektral.utils import convolution

batch_size = 10
N = 3
tol = 5.e-7


def _assert_all_close(output, expected_output):
    try:
        assert np.allclose(output, expected_output, atol=tol)
    except AssertionError:
        mean_diff = np.mean(np.absolute(output - expected_output))
        max_diff = np.max(np.absolute(output - expected_output))
        if mean_diff <= tol:
            print('Max difference above tolerance, but mean OK.')
        else:
            raise AssertionError('Mean diff: {}, Max diff: {}'.format(
                mean_diff, max_diff
            ))


def _convert_to_sparse_tensor(x):
    if x.ndim == 2:
        return ops.sp_matrix_to_sp_tensor(x)
    elif x.ndim == 3:
        s1_, s2_, s3_ = x.shape
        return ops.reshape(
            ops.sp_matrix_to_sp_tensor(x.reshape(s1_ * s2_, s3_)),
            (s1_, s2_, s3_)
        )


def _cast_all_to_dtype(values, dtype):
    return [tf.cast(v, dtype) for v in values]


def _check_op(op, numpy_inputs, expected_output, convert_to_sparse=None, **kwargs):
    output = _check_op_dense(op, numpy_inputs, **kwargs)
    _assert_all_close(output, expected_output)

    if convert_to_sparse:
        if isinstance(convert_to_sparse, list):
            if isinstance(convert_to_sparse[0], bool):
                # Make it into a list of list, always
                convert_to_sparse = [convert_to_sparse]

        for c_t_s in convert_to_sparse:
            output = _check_op_sparse(op, numpy_inputs, c_t_s, **kwargs)
            _assert_all_close(output, expected_output)


def _check_op_dense(op, numpy_inputs, **kwargs):
    tf_inputs = [tf.convert_to_tensor(x) for x in numpy_inputs]
    tf_inputs = _cast_all_to_dtype(tf_inputs, np.float32)

    output = op(*tf_inputs, **kwargs)
    if isinstance(output, tf.SparseTensor):
        # Sometimes ops with dense inputs return sparse tensors
        return tf.sparse.to_dense(output).numpy()
    return np.asarray(output)


def _check_op_sparse(op, numpy_inputs, convert_to_sparse, **kwargs):
    tf_inputs = []
    for i in range(len(numpy_inputs)):
        if convert_to_sparse[i]:
            tf_inputs.append(
                _convert_to_sparse_tensor(numpy_inputs[i])
            )
        else:
            tf_inputs.append(
                tf.convert_to_tensor(numpy_inputs[i])
            )
    tf_inputs = _cast_all_to_dtype(tf_inputs, np.float32)

    output = op(*tf_inputs, **kwargs)

    if hasattr(output, 'toarray'):
        return output.toarray()
    elif hasattr(output, 'numpy'):
        return output.numpy()
    elif isinstance(output, tf.SparseTensor):
        return tf.sparse.to_dense(output).numpy()
    else:
        return np.asarray(output)


def test_matmul_ops_single_mode():
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    _check_op(ops.matmul_A_B, [A, B], A.dot(B), convert_to_sparse)
    _check_op(ops.matmul_AT_B_A, [A, B], A.T.dot(B).dot(A), convert_to_sparse)
    _check_op(ops.matmul_AT_B, [A, B], A.T.dot(B), convert_to_sparse)
    _check_op(ops.matmul_A_BT, [A, B], A.dot(B.T), convert_to_sparse)


def test_matmul_ops_mixed_mode():
    A = np.random.randn(N, N)
    B = np.random.randn(batch_size, N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    # A * B
    expected_output = np.array([A.dot(B[i]) for i in range(batch_size)])
    _check_op(ops.matmul_A_B, [A, B], expected_output, convert_to_sparse)

    # A.T * B * A
    expected_output = np.array([A.T.dot(B[i]).dot(A) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B_A, [A, B], expected_output, convert_to_sparse)

    # A.T * B
    expected_output = np.array([A.T.dot(B[i]) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B, [A, B], expected_output, convert_to_sparse)

    # A * B.T
    expected_output = np.array([A.dot(B[i].T) for i in range(batch_size)])
    _check_op(ops.matmul_A_BT, [A, B], expected_output, convert_to_sparse)


def test_matmul_ops_inv_mixed_mode():
    A = np.random.randn(batch_size, N, N)
    B = np.random.randn(N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    # A * B
    expected_output = np.array([A[i].dot(B) for i in range(batch_size)])
    _check_op(ops.matmul_A_B, [A, B], expected_output, convert_to_sparse)

    # A.T * B * A
    expected_output = np.array([A[i].T.dot(B).dot(A[i]) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B_A, [A, B], expected_output, convert_to_sparse)

    # A.T * B
    expected_output = np.array([A[i].T.dot(B) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B, [A, B], expected_output, convert_to_sparse)

    # A * B.T
    expected_output = np.array([A[i].dot(B.T) for i in range(batch_size)])
    _check_op(ops.matmul_A_BT, [A, B], expected_output, convert_to_sparse)


def test_matmul_ops_batch_mode():
    A = np.random.randn(batch_size, N, N)
    B = np.random.randn(batch_size, N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    # A * B
    expected_output = np.array([A[i].dot(B[i]) for i in range(batch_size)])
    _check_op(ops.matmul_A_B, [A, B], expected_output, convert_to_sparse)

    # A.T * B * A
    expected_output = np.array([A[i].T.dot(B[i]).dot(A[i]) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B_A, [A, B], expected_output, convert_to_sparse)

    # A.T * B
    expected_output = np.array([A[i].T.dot(B[i]) for i in range(batch_size)])
    _check_op(ops.matmul_AT_B, [A, B], expected_output, convert_to_sparse)

    # A * B.T
    expected_output = np.array([A[i].dot(B[i].T) for i in range(batch_size)])
    _check_op(ops.matmul_A_BT, [A, B], expected_output, convert_to_sparse)


def test_graph_ops():
    A = np.ones((N, N))
    convert_to_sparse = [[True]]

    expected_output = convolution.normalized_adjacency(A)
    _check_op(ops.normalize_A, [A], expected_output, convert_to_sparse)

    expected_output = convolution.degree_matrix(A).sum(-1)
    _check_op(ops.degrees, [A], expected_output, convert_to_sparse)

    expected_output = convolution.degree_matrix(A)
    _check_op(ops.degree_matrix, [A], expected_output, convert_to_sparse)


def test_misc_ops():
    convert_to_sparse = [[True]]

    # Transpose
    for perm in [(1, 0), (0, 2, 1), (2, 1, 0)]:
        A = np.random.randn(*[N] * len(perm))
        expected_output = np.transpose(A, axes=perm)
        _check_op(ops.transpose, [A], expected_output, convert_to_sparse, perm=perm)

    # Reshape
    A = np.random.randn(4, 5)
    for shape in [(-1, 4), (5, -1)]:
        expected_output = np.reshape(A, shape)
        _check_op(ops.reshape, [A], expected_output, convert_to_sparse, shape=shape)

    # Matrix power
    A = np.random.randn(N, N)
    k = 4
    expected_output = np.linalg.matrix_power(A, k)
    _check_op(ops.matrix_power, [A], expected_output, convert_to_sparse, k=k)

    A = np.random.randn(batch_size, N, N)
    k = 4
    expected_output = np.array([np.linalg.matrix_power(a, k) for a in A])
    _check_op(ops.matrix_power, [A], expected_output, convert_to_sparse, k=k)


def test_modes_ops():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 2]])
    I = np.array([0, 0, 0, 1, 1])

    A_data = [1, 1, 1, 1, 1]
    A_row = [0, 1, 2, 3, 4]
    A_col = [1, 0, 1, 4, 3]
    A_sparse = coo_matrix((A_data, (A_row, A_col)), shape=(5, 5))
    A_sparse_tensor = ops.sp_matrix_to_sp_tensor(A_sparse)

    # Disjoint signal to batch
    expected_result = np.array([[[1., 0.],
                                 [0., 1.],
                                 [1., 1.]],
                                [[0., 0.],
                                 [1., 2.],
                                 [0., 0.]]])
    result = ops.disjoint_signal_to_batch(X, I).numpy()

    assert expected_result.shape == result.shape
    assert np.allclose(expected_result, result)

    # Disjoint adjacency to batch
    expected_result = np.array([[[0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 0.]],
                                [[0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 0., 0.]]])

    result = ops.disjoint_adjacency_to_batch(A_sparse_tensor, I).numpy()

    assert expected_result.shape == result.shape
    assert np.allclose(expected_result, result)
