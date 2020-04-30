import tensorflow as tf


def scatter_sum(updates, indices, N):
    """
    Sums updates along the first dimensions according to the indices, returns
    a Tensor of the same rank as updates with shape `(N, ...)`.
    If the result is empty for a given index `i`, `output[i] = 0`.
    If a given index`i` is negative, the value is ignored.
    :param updates: a Tensor.
    :param indices: A Tensor with indices to index the updates.
    :param N: first dimension the output (i.e., total number of segments).
    :return: a Tensor with the same rank as updates, of shape
    `(N, ) + updates.shape[1:]`.
    """
    return tf.math.unsorted_segment_sum(updates, indices, N)


def scatter_mean(updates, indices, N):
    """
    Averages updates along the first dimensions according to the indices,
    returns a Tensor of the same rank as updates with shape `(N, ...)`.
    If the result is empty for a given index `i`, `output[i] = 0`.
    If a given index`i` is negative, the value is ignored.
    :param updates: a Tensor.
    :param indices: A Tensor with indices to index the updates.
    :param N: first dimension the output (i.e., total number of segments).
    :return: a Tensor with the same rank as updates, of shape
    `(N, ) + updates.shape[1:]`.
    """
    return tf.math.unsorted_segment_mean(updates, indices, N)


# Alias for scatter_mean for convenience
scatter_avg = scatter_mean


def scatter_max(updates, indices, N):
    """
    Max-reduces updates along the first dimensions according to the indices,
    returns a Tensor of the same rank as updates with shape `(N, ...)`.
    If the result is empty for a given index `i`, `output[i] = 0`.
    If a given index`i` is negative, the value is ignored.
    :param updates: a Tensor.
    :param indices: A Tensor with indices to index the updates.
    :param N: first dimension the output (i.e., total number of segments).
    :return: a Tensor with the same rank as updates, of shape
    `(N, ) + updates.shape[1:]`.
    """
    return tf.math.unsorted_segment_max(updates, indices, N)


def scatter_min(updates, indices, N):
    """
    Min-reduces updates along the first dimensions according to the indices,
    returns a Tensor of the same rank as updates with shape `(N, ...)`.
    If the result is empty for a given index `i`, `output[i] = 0`.
    If a given index`i` is negative, the value is ignored.
    :param updates: a Tensor.
    :param indices: A Tensor with indices to index the updates.
    :param N: first dimension the output (i.e., total number of segments).
    :return: a Tensor with the same rank as updates, of shape
    `(N, ) + updates.shape[1:]`.
    """
    return tf.math.unsorted_segment_min(updates, indices, N)


def scatter_prod(updates, indices, N):
    """
    Multiplies updates along the first dimensions according to the indices,
    returns a Tensor of the same rank as updates with shape `(N, ...)`.
    If the result is empty for a given index `i`, `output[i] = 0`.
    If a given index`i` is negative, the value is ignored.
    :param updates: a Tensor.
    :param indices: A Tensor with indices to index the updates.
    :param N: first dimension the output (i.e., total number of segments).
    :return: a Tensor with the same rank as updates, of shape
    `(N, ) + updates.shape[1:]`.
    """
    return tf.math.unsorted_segment_prod(updates, indices, N)


OP_DICT = {
    'sum': scatter_sum,
    'mean': scatter_mean,
    'avg': scatter_avg,
    'max': scatter_max,
    'min': scatter_min,
    'prod': scatter_prod
}


def deserialize_scatter(scatter):
    if isinstance(scatter, str):
        if scatter in OP_DICT:
            return OP_DICT[scatter]
        else:
            if callable(scatter):
                return scatter
            else:
                raise ValueError('scatter must be callable or string in: {}.'
                                 .format(list(OP_DICT.keys())))
