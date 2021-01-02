import numpy as np


def pad_jagged_array(x, target_shape):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.
    :param x: a list or np.array of dtype object, containing np.arrays with
    variable dimensions;
    :param target_shape: a tuple or list s.t. target_shape[i] >= x.shape[i]
    for each x in X. If `target_shape[i] = -1`, it will be automatically
    converted to X.shape[i], so that passing a target shape of e.g. (-1, n, m)
    will leave the first  dimension of each element untouched.
    :return: a np.array of shape `(len(x), ) + target_shape`.
    """
    if len(x) < 1:
        raise ValueError("Jagged array cannot be empty")
    target_len = len(x)
    target_shape = tuple(
        shp if shp != -1 else x[0].shape[j] for j, shp in enumerate(target_shape)
    )
    output = np.zeros((target_len,) + target_shape, dtype=x[0].dtype)
    for i in range(target_len):
        slc = (i,) + tuple(slice(shp) for shp in x[i].shape)
        output[slc] = x[i]

    return output


def one_hot(x, depth):
    """
    One-hot encodes the integer array `x` in an array of length `depth`.
    :param x: a np.array of integers.
    :param depth: size of the one-hot vectors.
    :return: an array of shape `x.shape + (depth, )`
    """
    x = np.array(x).astype(int)
    out = np.eye(depth)[x]

    return out


def label_to_one_hot(x, labels):
    """
    One-hot encodes the integer array `x` according to the given `labels`.

    :param x: a np.array of integers. Each value must be contained in `labels`.
    :param labels: list/tuple/np.array of labels.
    :return: an array of shape `x.shape + (len(labels), )`
    """
    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise ValueError("labels must be list, tuple, or np.ndarray")
    if not np.all(np.in1d(x, labels)):
        raise ValueError("All values in x must be contained in labels")
    depth = len(labels)
    x = np.array(x).astype(int)
    out = x.copy()
    for i, label in enumerate(labels):
        out[x == label] = i

    return one_hot(out, depth)


def _flatten_list_gen(alist):
    """
    Performs a depth-first visit of an arbitrarily nested list and yields its
    element in order.
    :param alist: a list or np.array (with at least one dimension),
                  arbitrarily nested.
    """
    for item in alist:
        if isinstance(item, (list, tuple, np.ndarray)):
            for i in _flatten_list_gen(item):
                yield i
        else:
            yield item


def flatten_list(alist):
    """
    Flattens an arbitrarily nested list to 1D.
    :param alist: a list or np.array (with at least one dimension),
                  arbitrarily nested.
    :return: a 1D Python list with the flattened elements as returned by a
             depth-first search.
    """
    return list(_flatten_list_gen(alist))
