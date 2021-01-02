import tensorflow as tf


def mixed_mode_support(scatter_fn):
    def _wrapper_mm_support(updates, indices, N):
        if len(tf.shape(updates)) == 3:
            updates = tf.transpose(updates, perm=(1, 0, 2))
        out = scatter_fn(updates, indices, N)
        if len(tf.shape(out)) == 3:
            out = tf.transpose(out, perm=(1, 0, 2))
        return out

    _wrapper_mm_support.__name__ = scatter_fn.__name__
    return _wrapper_mm_support


@mixed_mode_support
def scatter_sum(messages, indices, n_nodes):
    """
    Sums messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all zeros.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_sum(messages, indices, n_nodes)


@mixed_mode_support
def scatter_mean(messages, indices, n_nodes):
    """
    Averages messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all zeros.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_mean(messages, indices, n_nodes)


@mixed_mode_support
def scatter_max(messages, indices, n_nodes):
    """
    Max-reduces messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be the minimum possible
    value for the dtype of the messages
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_max(messages, indices, n_nodes)


@mixed_mode_support
def scatter_min(messages, indices, n_nodes):
    """
    Min-reduces messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be the maximum possible
    value for the dtype of the messages.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_min(messages, indices, n_nodes)


@mixed_mode_support
def scatter_prod(messages, indices, n_nodes):
    """
    Multiplies messages element-wise according to the segments defined by
    `indices`, with support for messages in single/disjoint mode (rank 2)
    and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all ones.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_prod(messages, indices, n_nodes)


OP_DICT = {
    "sum": scatter_sum,
    "mean": scatter_mean,
    "max": scatter_max,
    "min": scatter_min,
    "prod": scatter_prod,
}


def unsorted_segment_softmax(x, indices, n_nodes=None):
    """
    Applies softmax along the segments of a Tensor. This operator is similar
    to the tf.math.segment_* operators, which apply a certain reduction to the
    segments. In this case, the output tensor is not reduced and maintains the
    same shape as the input.
    :param x: a Tensor. The softmax is applied along the first dimension.
    :param indices: a Tensor, indices to the segments.
    :param n_nodes: the number of unique segments in the indices. If `None`,
    n_nodes is calculated as the maximum entry in the indices plus 1.
    :return: a Tensor with the same shape as the input.
    """
    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
    e_x = tf.exp(
        x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
    )
    e_x /= tf.gather(
        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
    )
    return e_x


def serialize_scatter(identifier):
    if identifier in OP_DICT:
        return identifier
    elif hasattr(identifier, "__name__"):
        for k, v in OP_DICT.items():
            if v.__name__ == identifier.__name__:
                return k
        return None


def deserialize_scatter(scatter):
    if isinstance(scatter, str):
        if scatter in OP_DICT:
            return OP_DICT[scatter]
        else:
            if callable(scatter):
                return scatter
            else:
                raise ValueError(
                    "scatter must be callable or string in: {}.".format(
                        list(OP_DICT.keys())
                    )
                )
