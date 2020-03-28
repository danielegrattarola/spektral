import tensorflow as tf


def scatter_sum(indices, updates, N):
    return tf.math.unsorted_segment_sum(indices, updates, N)


def scatter_mean(indices, updates, N):
    return tf.math.unsorted_segment_mean(indices, updates, N)


def scatter_max(indices, updates, N):
    return tf.math.unsorted_segment_max(indices, updates, N)


def scatter_min(indices, updates, N):
    return tf.math.unsorted_segment_min(indices, updates, N)


def scatter_prod(indices, updates, N):
    return tf.math.unsorted_segment_prod(indices, updates, N)
