from __future__ import division

import re

import numpy as np
from scipy import sparse as sp


def pad_jagged_array(x, target_shape, dtype=np.float):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.
    :param x: a np.array of dtype object, containing np.arrays of varying 
    dimensions
    :param target_shape: a tuple or list s.t. target_shape[i] >= x.shape[i]
    for each x in X.
    If `target_shape[i] = -1`, it will be automatically converted to X.shape[i], 
    so that passing a target shape of e.g. (-1, n, m) will leave the first 
    dimension of each element untouched (note that the creation of the output
    array may fail if the result is again a jagged array). 
    :param dtype: the dtype of the returned np.array
    :return: a zero-padded np.array of shape `(X.shape[0], ) + target_shape`
    """
    for i in range(len(x)):
        shapes = []
        for j in range(len(target_shape)):
            ts = target_shape[j]
            cs = x[i].shape[j]
            shapes.append((cs if ts == -1 else ts, cs))
        if x.ndim == 1:
            x[i] = np.pad(x[i], [(0, ts - cs) for ts, cs in shapes], 'constant')
        else:
            x = np.pad(x, [(0, 0)] + [(0, ts - cs) for ts, cs in shapes], 'constant')

    try:
        return np.array(x, dtype=dtype)
    except ValueError:
        return np.array([_ for _ in x], dtype=dtype)


def normalize_sum_to_unity(x):
    """
    Normalizes each row of the input to have a sum of 1.
    :param x: np.array or scipy.sparse matrix of shape `(num_nodes, num_nodes)`
    or `(batch, num_nodes, num_nodes)`
    :return: np.array of the same shape as x 
    """
    if x.ndim == 3:
        return np.nan_to_num(x / x.sum(-1)[..., np.newaxis])
    else:
        return np.nan_to_num(x / x.sum(-1).reshape(-1, 1))


def add_eye(x):
    """
    Adds the identity matrix to the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array as described above
    """
    if x.ndim != 2:
        raise ValueError('X must be of rank 2 but has rank {}.'.format(x.ndim))
    if sp.issparse(x):
        eye = sp.eye(x.shape[0])
    else:
        eye = np.eye(x.shape[0])
    return x + eye


def sub_eye(x):
    """
    Subtracts the identity matrix to the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array as described above
    """
    if x.ndim != 2:
        raise ValueError('x must be of rank 2 but has rank {}.'.format(x.ndim))
    if sp.issparse(x):
        eye = sp.eye(x.shape[0])
    else:
        eye = np.eye(x.shape[0])
    return x - eye


def add_eye_batch(x):
    """
    Adds the identity matrix to each 2D slice of the given 3D array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array as described above
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x + np.eye(x.shape[1])[None, ...]


def sub_eye_batch(x):
    """
    Subtracts the identity matrix from each 2D slice of the given 3D array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array as described above
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x - np.repeat(np.eye(x.shape[1])[None, ...], x.shape[0], axis=0)


def add_eye_jagged(x):
    """
    Adds the identity matrix to each 2D element of the given 3D jagged array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array as described above
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = add_eye(x[i])
    return x_out


def sub_eye_jagged(x):
    """
    Subtracts the identity matrix to each 2D element of the given 3D jagged 
    array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array as described above
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = sub_eye(x[i])
    return x_out


def natural_key(string_):
    """
    Key function for natural sorting using the `sorted` builtin.
    :param string_: a string
    :return: a rearrangement of the string s.t. sorting a list of strings with 
    this function as key results in natural sorting
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def int_to_one_hot(x, n=None):
    """
    Encodes x in a 1-of-n array. 
    :param x: an integer or array of integers, such that x < n
    :param n: an integer
    :return: an array of shape (x.shape[0], n) if x is an array, (n, ) if
    x is an integer
    """
    if isinstance(x, int):
        if n is None:
            raise ValueError('n is required to one-hot encode a single integer')
        if x >= n:
            raise ValueError('x must be smaller than n in order to one-hot encode')
        output = np.zeros((n,))
        output[x] = 1
    else:
        if n is None:
            n = int(np.max(x) + 1)
        else:
            if np.max(x) >= n:
                raise ValueError('The maximum value in x ({}) is greater than '
                                 'n ({}), therefore 1-of-n encoding is not '
                                 'possible'.format(np.max(x), n))
        x = np.array(x, dtype=np.int)
        if x.ndim is 1:
            x = x[:, None]
        orig_shp = x.shape
        x = np.reshape(x, (-1, orig_shp[-1]))
        output = np.zeros((x.shape[0], n))
        output[np.arange(x.shape[0]), x.squeeze()] = 1
        output = output.reshape(orig_shp[:-1] + (n,))

    return output


def label_to_one_hot(x, labels=None):
    """
    Encodes x in a 1-of-n array. 
    :param x: any object or array of objects s.t. x is contained in `labels`. 
    The function may behave unexpectedly if x is a single object but 
    `hasattr(x, '__len__')`, and works best with integers or discrete entities.
    :param labels: a list of n labels to compute the one-hot vector 
    :return: an array of shape (x.shape[0], n) if x is an array, (n, ) if
    x is a single object
    """
    n = len(labels)
    labels_idx = {l: i for i, l in enumerate(labels)}
    if not hasattr(x, '__len__'):
        output = np.zeros((n,))
        output[labels_idx[x]] = 1
    else:
        x = np.array(x, dtype=np.int)
        orig_shp = x.shape
        x = np.reshape(x, (-1))
        x = np.array([labels_idx[_] for _ in x])
        output = np.zeros((x.shape[0], n))
        output[np.arange(x.shape[0]), x] = 1
        if len(orig_shp) == 1:
            output_shape = orig_shp + (n,)
        else:
            output_shape = orig_shp[:-1] + (n,)
        output = output.reshape(output_shape)

    return output


def int_to_one_hot_closure(n):
    """
    Retruns a function with signature `foo(x)` equivalent to calling 
    `int_to_one_hot(x, n=n)`. This is especially useful when using 
    `int_to_one_hot` as preprocessing function as it would be impossible to 
    manually assign `n` at evey call (see `utils.datasets.molecules` for an 
    example of usage).
    :param n: see the `int_to_one_hot` documentation
    :return: a function with signature foo(x)
    """

    def int_to_one_hot_fun(x):
        return int_to_one_hot(x, n=n)

    return int_to_one_hot_fun


def label_to_one_hot_closure(labels):
    """
    Retruns a function with signature `foo(x)` equivalent to calling 
    `label_to_one_hot(x, labels=labels)`. This is especially useful when using 
    `label_to_one_hot` as preprocessing function as it would be impossible to 
    manually assign `labels` at evey call (see `utils.datasets.molecules` for an 
    example of usage).
    :param labels: see the `label_to_one_hot` documentation
    :return: a function with signature foo(x)
    """

    def label_to_one_hot_fun(x):
        return label_to_one_hot(x, labels=labels)

    return label_to_one_hot_fun


def idx_to_mask(idx, shape):
    """
    Creates a boolean mask with the given shape in which the elements at idx
    are True.
    :param idx: a list or np.array of integer indices 
    :param shape: a tuple representing the mask shape
    :return: a boolean np.array  
    """
    output = np.zeros(shape)
    output[idx] = 1
    return output.astype(np.bool)


def flatten_list_gen(alist):
    """
    Performs a depth-first visit of an arbitrarily nested list and yields its 
    element in order. 
    :param alist: a list or np.array (with at least one dimension), 
                  arbitrarily nested.
    """
    for item in alist:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            for i in flatten_list_gen(item):
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
    return list(flatten_list_gen(alist))


def batch_iterator(data, batch_size=32, epochs=1, shuffle=True):
    """
    Iterates over the data for the given number of epochs, yielding batches of
    size `batch_size`.
    :param data: np.array or list of np.arrays with equal first dimension.
    :param batch_size: number of samples in a batch
    :param epochs: number of times to iterate over the data
    :param shuffle: whether to shuffle the data at the beginning of each epoch
    :yield: a batch of samples (or tuple of batches if X had more than one 
    array). 
    """
    if not isinstance(data, list):
        data = [data]
    if len(set([len(item) for item in data])) > 1:
        raise ValueError('All arrays must have the same length')

    len_data = len(data[0])
    batches_per_epoch = int(len_data / batch_size)
    if len_data % batch_size != 0:
        batches_per_epoch += 1
    for epochs in range(epochs):
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(len_data))
            data = [np.array(item)[shuffle_idx] for item in data]
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            if len(data) > 1:
                yield [item[start:stop] for item in data]
            else:
                yield data[0][start:stop]


def set_trainable(model, toset):
    """
    Sets the trainable parameters of a Keras model and all its layers to toset.
    :param model: a Keras Model
    :param toset: boolean
    :return: None
    """
    for layer in model.layers:
        layer.trainable = toset
    model.trainable = toset
