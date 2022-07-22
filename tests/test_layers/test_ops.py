import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from spektral.data.utils import to_disjoint
from spektral.layers import ops
from spektral.utils import convolution
from spektral.utils.sparse import sp_batch_to_sp_tensor, sp_matrix_to_sp_tensor

batch_size = 10
N = 3
M = 5
tol = 1e-5


def _assert_all_close(output, expected_output):
    try:
        assert np.allclose(output, expected_output, atol=tol)
    except AssertionError as ae:
        mean_diff = np.mean(np.absolute(output - expected_output))
        max_diff = np.max(np.absolute(output - expected_output))
        if mean_diff <= tol:
            print("Max difference above tolerance, but mean OK.")
        else:
            raise AssertionError(
                "Mean diff: {}, Max diff: {}".format(mean_diff, max_diff)
            ) from ae


def _convert_to_sparse_tensor(x):
    if x.ndim == 2:
        return sp_matrix_to_sp_tensor(x)
    elif x.ndim == 3:
        s1_, s2_, s3_ = x.shape
        return ops.reshape(
            sp_matrix_to_sp_tensor(x.reshape(s1_ * s2_, s3_)), (s1_, s2_, s3_)
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
            tf_inputs.append(_convert_to_sparse_tensor(numpy_inputs[i]))
        else:
            tf_inputs.append(tf.convert_to_tensor(numpy_inputs[i]))
    tf_inputs = _cast_all_to_dtype(tf_inputs, np.float32)

    output = op(*tf_inputs, **kwargs)

    if hasattr(output, "toarray"):
        return output.toarray()
    elif hasattr(output, "numpy"):
        return output.numpy()
    elif isinstance(output, tf.SparseTensor):
        return tf.sparse.to_dense(output).numpy()
    else:
        return np.asarray(output)


def test_matmul_ops_rank_2():
    a = np.random.randn(N, N)
    b = np.random.randn(N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    _check_op(ops.modal_dot, [a, b], a.dot(b), convert_to_sparse)
    _check_op(ops.modal_dot, [a, b], a.T.dot(b), convert_to_sparse, transpose_a=True)
    _check_op(ops.modal_dot, [a, b], a.dot(b.T), convert_to_sparse, transpose_b=True)
    _check_op(
        ops.modal_dot,
        [a, b],
        a.T.dot(b.T),
        convert_to_sparse,
        transpose_a=True,
        transpose_b=True,
    )
    _check_op(ops.matmul_at_b_a, [a, b], a.T.dot(b).dot(a), convert_to_sparse)


def test_matmul_ops_rank_2_3():
    a = np.random.randn(N, N)
    b = np.random.randn(batch_size, N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    expected_output = np.array([a.dot(b[i]) for i in range(batch_size)])
    _check_op(ops.modal_dot, [a, b], expected_output, convert_to_sparse)

    expected_output = np.array([a.T.dot(b[i]) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_a=True
    )

    expected_output = np.array([a.dot(b[i].T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_b=True
    )

    expected_output = np.array([a.T.dot(b[i].T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot,
        [a, b],
        expected_output,
        convert_to_sparse,
        transpose_a=True,
        transpose_b=True,
    )

    expected_output = np.array([a.T.dot(b[i]).dot(a) for i in range(batch_size)])
    _check_op(ops.matmul_at_b_a, [a, b], expected_output, convert_to_sparse)

    # Check with outer dimensions that don't match.
    c = np.random.randn(M, N)
    expected_output = np.array([c.dot(b[i]) for i in range(batch_size)])
    _check_op(ops.modal_dot, [c, b], expected_output, convert_to_sparse)


def test_matmul_ops_rank_3_2():
    a = np.random.randn(batch_size, N, N)
    b = np.random.randn(N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    expected_output = np.array([a[i].dot(b) for i in range(batch_size)])
    _check_op(ops.modal_dot, [a, b], expected_output, convert_to_sparse)

    expected_output = np.array([a[i].T.dot(b) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_a=True
    )

    expected_output = np.array([a[i].dot(b.T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_b=True
    )

    expected_output = np.array([a[i].T.dot(b.T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot,
        [a, b],
        expected_output,
        convert_to_sparse,
        transpose_a=True,
        transpose_b=True,
    )

    expected_output = np.array([a[i].T.dot(b).dot(a[i]) for i in range(batch_size)])
    _check_op(ops.matmul_at_b_a, [a, b], expected_output, convert_to_sparse)


def test_matmul_ops_rank_3():
    a = np.random.randn(batch_size, N, N)
    b = np.random.randn(batch_size, N, N)
    convert_to_sparse = [[True, False], [False, True], [True, True]]

    expected_output = np.array([a[i].dot(b[i]) for i in range(batch_size)])
    _check_op(ops.modal_dot, [a, b], expected_output, convert_to_sparse)

    expected_output = np.array([a[i].T.dot(b[i]) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_a=True
    )

    expected_output = np.array([a[i].dot(b[i].T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot, [a, b], expected_output, convert_to_sparse, transpose_b=True
    )

    expected_output = np.array([a[i].T.dot(b[i].T) for i in range(batch_size)])
    _check_op(
        ops.modal_dot,
        [a, b],
        expected_output,
        convert_to_sparse,
        transpose_a=True,
        transpose_b=True,
    )

    expected_output = np.array([a[i].T.dot(b[i]).dot(a[i]) for i in range(batch_size)])
    _check_op(ops.matmul_at_b_a, [a, b], expected_output, convert_to_sparse)


def test_graph_ops():
    A = np.ones((N, N))
    convert_to_sparse = [[True]]

    expected_output = convolution.normalized_adjacency(A)
    _check_op(ops.normalize_A, [A], expected_output, convert_to_sparse)

    expected_output = convolution.degree_matrix(A).sum(-1)
    _check_op(ops.degrees, [A], expected_output, convert_to_sparse)

    expected_output = convolution.degree_matrix(A)
    _check_op(ops.degree_matrix, [A], expected_output, convert_to_sparse)


def test_base_ops():
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

    # Repeat
    X = np.random.rand(N)
    repeats = np.random.randint(1, 4, N)
    expected_output = np.repeat(X, repeats)
    _check_op(ops.repeat, [X], expected_output, repeats=repeats)

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
    ns = [10, 5, 8]
    x_list = [np.random.randn(n, 3) for n in ns]
    a_list = [sp.csr_matrix(np.ones((n, n))) for n in ns]

    x, a, i = to_disjoint(x_list, a_list)
    a = sp_matrix_to_sp_tensor(a)

    expected_x = np.zeros((len(ns), max(ns), 3))
    for i_, n in enumerate(ns):
        expected_x[i_, :n] = x_list[i_]

    expected_a = np.zeros((len(ns), max(ns), max(ns)))
    for i_, n in enumerate(ns):
        expected_a[i_, :n, :n] = a_list[i_].A

    # Disjoint signal to batch
    result = ops.disjoint_signal_to_batch(x, i).numpy()

    assert expected_x.shape == result.shape
    assert np.allclose(expected_x, result, atol=tol)

    # Disjoint adjacency to batch
    result = ops.disjoint_adjacency_to_batch(a, i).numpy()

    assert expected_a.shape == result.shape
    assert np.allclose(expected_a, result, atol=tol)


def test_scatter_ops():
    from spektral.layers.ops.scatter import OP_DICT

    indices = np.array([0, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    n_messages = len(indices)
    n_nodes = indices.max() + 1
    n_features = 10
    batch_size = 3
    messages = np.ones((n_messages, n_features))
    messages_mixed = np.array([messages] * batch_size)
    messages_random = np.random.rand(batch_size, n_messages, n_features)
    NO_INDEX_VAL = {
        "sum": 0.0,
        "mean": 0.0,
        "max": tf.as_dtype(messages.dtype).min,
        "min": tf.as_dtype(messages.dtype).max,
        "prod": 1.0,
    }

    for key, scatter_fn in OP_DICT.items():
        # Test serialization
        assert ops.deserialize_scatter(key) == scatter_fn
        assert ops.serialize_scatter(scatter_fn) == key

        # Test single mode
        out = scatter_fn(messages, indices, n_nodes)
        assert out.shape == (n_nodes, n_features)
        assert np.all(out[3] == NO_INDEX_VAL[key])
        if key == "sum":
            assert np.all(
                out == np.tile([1, 2, 3, NO_INDEX_VAL[key], 4], [n_features, 1]).T
            )
        else:
            assert np.all(
                out == np.tile([1, 1, 1, NO_INDEX_VAL[key], 1], [n_features, 1]).T
            )

        # Test batch mode
        out = scatter_fn(messages_mixed, indices, n_nodes)
        assert out.shape == (batch_size, n_nodes, n_features)
        assert np.all(out[:, 3, :] == NO_INDEX_VAL[key])
        for i in range(batch_size):
            if key == "sum":
                assert np.all(
                    out[i]
                    == np.tile([1, 2, 3, NO_INDEX_VAL[key], 4], [n_features, 1]).T
                )
            else:
                assert np.all(
                    out[i]
                    == np.tile([1, 1, 1, NO_INDEX_VAL[key], 1], [n_features, 1]).T
                )

        # Test equivalence on random inputs
        out_mixed = scatter_fn(messages_random, indices, n_nodes)
        for i in range(batch_size):
            assert np.allclose(
                out_mixed[i], scatter_fn(messages_random[i], indices, n_nodes), atol=tol
            )


def test_segment_top_k():
    x = np.array([0.2, 0.5, 0.3, -0.1, -0.2, -0.1], dtype=np.float32)
    I = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    ratio = 0.5
    topk = ops.segment_top_k(x, I, ratio)
    actual = topk.numpy()
    expected = [1, 2, 5]
    np.testing.assert_equal(actual, expected)


def test_indices_to_mask_rank1():
    indices = [1, 3, 4]
    mask = ops.indices_to_mask(indices, 6)
    np.testing.assert_equal(mask.numpy(), [0, 1, 0, 1, 1, 0])


def test_indices_to_mask_rank2():
    indices = [[0, 2], [1, 1], [2, 1]]
    mask = ops.indices_to_mask(indices, [3, 3])
    expected = [[0, 0, 1], [0, 1, 0], [0, 1, 0]]
    np.testing.assert_equal(mask.numpy(), expected)


def random_sparse(shape, nnz):
    max_index = np.prod(shape)
    indices = np.random.choice(max_index, nnz, replace=False)
    indices.sort()
    indices = np.stack(np.unravel_index(indices, shape), axis=-1)
    return tf.SparseTensor(indices, np.random.normal(size=(nnz,)), shape)


def test_boolean_mask_sparse():
    st = random_sparse((5, 5), 15)
    dense = tf.sparse.to_dense(st)
    mask = np.array([0, 1, 0, 1, 1], dtype=np.bool)
    for axis in (0, 1):
        actual, _ = ops.boolean_mask_sparse(st, mask, axis=axis)
        actual = tf.sparse.to_dense(actual).numpy()
        expected = tf.boolean_mask(dense, mask, axis=axis).numpy()
        np.testing.assert_equal(actual, expected)


def test_boolean_mask_sparse_square():
    st = random_sparse((5, 5), 15)
    dense = tf.sparse.to_dense(st)
    mask = np.array([0, 1, 0, 1, 1], dtype=np.bool)
    actual, _ = ops.boolean_mask_sparse_square(st, mask)
    for axis in (0, 1):
        dense = tf.boolean_mask(dense, mask, axis=axis)
    actual = tf.sparse.to_dense(actual)
    np.testing.assert_equal(actual.numpy(), dense.numpy())


def test_gather_sparse():
    st = random_sparse((5, 5), 15)
    dense = tf.sparse.to_dense(st)
    indices = np.array([1, 3, 4], dtype=np.int64)
    for axis in (0, 1):
        actual, _ = ops.gather_sparse(st, indices, axis=axis)
        actual = tf.sparse.to_dense(actual).numpy()
        expected = tf.gather(dense, indices, axis=axis).numpy()
        np.testing.assert_equal(actual, expected)


def test_gather_sparse_square():
    st = random_sparse((5, 5), 15)
    dense = tf.sparse.to_dense(st)
    indices = np.array([1, 3, 4], dtype=np.int64)
    actual, _ = ops.gather_sparse_square(st, indices)
    for axis in (0, 1):
        dense = tf.gather(dense, indices, axis=axis)
    actual = tf.sparse.to_dense(actual)
    np.testing.assert_equal(actual.numpy(), dense.numpy())


def test_sp_batch_to_sp_tensor():
    a_batch = [sp.random(N, N) for _ in range(batch_size)]
    a_out = sp_batch_to_sp_tensor(a_batch)
    assert a_out.shape == (batch_size, N, N)
