from spektral.layers.ops import matmul_A_B, matmul_AT_B, matmul_A_BT, matmul_AT_B_A, sp_matrix_to_sp_tensor, reshape
import numpy as np
from keras import backend as K

sess = K.get_session()
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
        return sp_matrix_to_sp_tensor(x)
    elif x.ndim == 3:
        s1_, s2_, s3_ = x.shape
        return reshape(
            sp_matrix_to_sp_tensor(x.reshape(s1_ * s2_, s3_)),
            (s1_, s2_, s3_)
        )


def _cast_all_to_dtype(values, dtype):
    return [K.cast(v, dtype) for v in values]


def _check_op(op, numpy_inputs, expected_output, convert_to_sparse=None):
    output = _check_op_dense(op, numpy_inputs)
    _assert_all_close(output, expected_output)

    if convert_to_sparse:
        if isinstance(convert_to_sparse, list):
            if isinstance(convert_to_sparse[0], bool):
                # Make it into a list of list, always
                convert_to_sparse = [convert_to_sparse]

        for c_t_s in convert_to_sparse:
            output = _check_op_sparse(op, numpy_inputs, c_t_s)
            _assert_all_close(output, expected_output)


def _check_op_dense(op, numpy_inputs):
    tf_inputs = [K.constant(x) for x in numpy_inputs]
    tf_inputs = _cast_all_to_dtype(tf_inputs, np.float32)

    op_result = op(*tf_inputs)
    output = sess.run(op_result)

    return np.asarray(output)


def _check_op_sparse(op, numpy_inputs, convert_to_sparse):
    tf_inputs = []
    for i in range(len(numpy_inputs)):
        if convert_to_sparse[i]:
            tf_inputs.append(
                _convert_to_sparse_tensor(numpy_inputs[i])
            )
        else:
            tf_inputs.append(
                K.constant(numpy_inputs[i])
            )
    tf_inputs = _cast_all_to_dtype(tf_inputs, np.float32)

    op_result = op(*tf_inputs)
    output = sess.run(op_result)

    if hasattr(output, 'toarray'):
        return output.toarray()
    else:
        return np.asarray(output)


def test_matmul_ops_single_mode():
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)
    convert_to_sparse = [[True, False], [False, True]]
    _check_op(matmul_A_B, [A, B], A.dot(B), convert_to_sparse)
    _check_op(matmul_AT_B_A, [A, B], A.T.dot(B).dot(A), convert_to_sparse)
    _check_op(matmul_AT_B, [A, B], A.T.dot(B), convert_to_sparse)
    _check_op(matmul_A_BT, [A, B], A.dot(B.T), convert_to_sparse)


def test_matmul_ops_mixed_mode():
    A = np.random.randn(N, N)
    B = np.random.randn(batch_size, N, N)
    convert_to_sparse = [[True, False], [False, True]]

    # A * B
    expected_output = np.array([A.dot(B[i]) for i in range(batch_size)])
    _check_op(matmul_A_B, [A, B], expected_output, convert_to_sparse)

    # A.T * B * A
    expected_output = np.array([A.T.dot(B[i]).dot(A) for i in range(batch_size)])
    _check_op(matmul_AT_B_A, [A, B], expected_output, convert_to_sparse)

    # A.T * B
    expected_output = np.array([A.T.dot(B[i]) for i in range(batch_size)])
    _check_op(matmul_AT_B, [A, B], expected_output, convert_to_sparse)

    # A * B.T
    expected_output = np.array([A.dot(B[i].T) for i in range(batch_size)])
    _check_op(matmul_A_BT, [A, B], expected_output, convert_to_sparse)


def test_matmul_ops_inv_mixed_mode():
    A = np.random.randn(batch_size, N, N)
    B = np.random.randn(N, N)

    # A * B
    expected_output = np.array([A[i].dot(B) for i in range(batch_size)])
    _check_op(matmul_A_B, [A, B], expected_output)

    # A.T * B * A
    expected_output = np.array([A[i].T.dot(B).dot(A[i]) for i in range(batch_size)])
    _check_op(matmul_AT_B_A, [A, B], expected_output)

    # A.T * B
    expected_output = np.array([A[i].T.dot(B) for i in range(batch_size)])
    _check_op(matmul_AT_B, [A, B], expected_output)

    # A * B.T
    expected_output = np.array([A[i].dot(B.T) for i in range(batch_size)])
    _check_op(matmul_A_BT, [A, B], expected_output)


def test_matmul_ops_batch_mode():
    A = np.random.randn(batch_size, N, N)
    B = np.random.randn(batch_size, N, N)

    # A * B
    expected_output = np.array([A[i].dot(B[i]) for i in range(batch_size)])
    _check_op(matmul_A_B, [A, B], expected_output)

    # A.T * B * A
    expected_output = np.array([A[i].T.dot(B[i]).dot(A[i]) for i in range(batch_size)])
    _check_op(matmul_AT_B_A, [A, B], expected_output)

    # A.T * B
    expected_output = np.array([A[i].T.dot(B[i]) for i in range(batch_size)])
    _check_op(matmul_AT_B, [A, B], expected_output)

    # A * B.T
    expected_output = np.array([A[i].dot(B[i].T) for i in range(batch_size)])
    _check_op(matmul_A_BT, [A, B], expected_output)
