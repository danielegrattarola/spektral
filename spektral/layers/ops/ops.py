import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def transpose(a, perm=None, name=None):
    """
    Transposes a according to perm, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(a, perm=perm, name=name)


def reshape(a, shape=None, name=None):
    """
    Reshapes a according to shape, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)


def repeat(x, repeats):
    """
    Repeats elements of a Tensor (equivalent to np.repeat, but only for 1D
    tensors).
    :param x: rank 1 Tensor;
    :param repeats: rank 1 Tensor with same shape as x, the number of
    repetitions for each element;
    :return: rank 1 Tensor, of shape `(sum(repeats), )`.
    """
    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def segment_top_k(x, I, ratio):
    """
    Returns indices to get the top K values in x segment-wise, according to
    the segments defined in I. K is not fixed, but it is defined as a ratio of
    the number of elements in each segment.
    :param x: a rank 1 Tensor;
    :param I: a rank 1 Tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :return: a rank 1 Tensor containing the indices to get the top K values of
    each segment in x.
    """
    rt = tf.RaggedTensor.from_value_rowids(x, I)
    row_lengths = rt.row_lengths()
    dense = rt.to_tensor(default_value=-np.inf)
    indices = tf.cast(tf.argsort(dense, direction="DESCENDING"), tf.int64)
    row_starts = tf.cast(rt.row_starts(), tf.int64)
    indices = indices + tf.expand_dims(row_starts, 1)
    row_lengths = tf.cast(
        tf.math.ceil(ratio * tf.cast(row_lengths, tf.float32)), tf.int32
    )
    return tf.RaggedTensor.from_tensor(indices, row_lengths).values


def indices_to_mask(indices, shape, dtype=tf.bool):
    """
    Return mask with true values at indices of the given shape.
    This can be used as an inverse to tf.where.
    :param indices: [nnz, k] or [nnz] Tensor indices of True values.
    :param shape: [k] or [] (scalar) Tensor shape/size of output.
    :param dtype: dtype of the output.
    :return: Tensor of given shape and dtype.
    """
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    if indices.shape.ndims == 1:
        assert isinstance(shape, int) or shape.shape.ndims == 0
        indices = tf.expand_dims(indices, axis=1)
        if isinstance(shape, int):
            shape = tf.TensorShape([shape])
        else:
            shape = tf.expand_dims(shape, axis=0)
    else:
        indices.shape.assert_has_rank(2)
    assert indices.dtype.is_integer
    nnz = tf.shape(indices)[0]
    indices = tf.cast(indices, tf.int64)
    shape = tf.cast(shape, tf.int64)
    return tf.scatter_nd(indices, tf.ones((nnz,), dtype=dtype), shape)
