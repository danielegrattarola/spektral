import numpy as np
from scipy import sparse as sp


def pad_jagged_array(x, target_shape, dtype=np.float):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.
    :param x: a list or np.array of dtype object, containing np.arrays of
    varying dimensions
    :param target_shape: a tuple or list s.t. target_shape[i] >= x.shape[i]
    for each x in X.
    If `target_shape[i] = -1`, it will be automatically converted to X.shape[i], 
    so that passing a target shape of e.g. (-1, n, m) will leave the first 
    dimension of each element untouched (note that the creation of the output
    array may fail if the result is again a jagged array). 
    :param dtype: the dtype of the returned np.array
    :return: a zero-padded np.array of shape `(X.shape[0], ) + target_shape`
    """
    if isinstance(x, list):
        x = np.array(x)
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


def add_eye(x):
    """
    Adds the identity matrix to the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array or scipy.sparse matrix
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
    Subtracts the identity matrix from the given matrix.
    :param x: a rank 2 np.array or scipy.sparse matrix
    :return: a rank 2 np.array or scipy.sparse matrix
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
    Adds the identity matrix to each submatrix of the given rank 3 array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x + np.eye(x.shape[1])[None, ...]


def sub_eye_batch(x):
    """
    Subtracts the identity matrix from each submatrix of the given rank 3
    array.
    :param x: a rank 3 np.array
    :return: a rank 3 np.array
    """
    if x.ndim != 3:
        raise ValueError('x must be of rank 3 but has rank {}.'.format(x.ndim))
    return x - np.repeat(np.eye(x.shape[1])[None, ...], x.shape[0], axis=0)


def add_eye_jagged(x):
    """
    Adds the identity matrix to each submatrix of the given rank 3 jagged array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = add_eye(x[i])
    return x_out


def sub_eye_jagged(x):
    """
    Subtracts the identity matrix from each submatrix of the given rank 3
    jagged array.
    :param x: a rank 3 jagged np.array
    :return: a rank 3 jagged np.array
    """
    x_out = x.copy()
    for i in range(len(x)):
        if x[i].ndim != 2:
            raise ValueError('Jagged array must only contain 2d slices')
        x_out[i] = sub_eye(x[i])
    return x_out


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
        if x.ndim == 1:
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
        output = np.zeros((x.shape[0], n))
        for i in range(len(x)):
            try:
                output[i, labels_idx[x[i]]] = 1
            except KeyError:
                pass
        if len(orig_shp) == 1:
            output_shape = orig_shp + (n,)
        else:
            output_shape = orig_shp[:-1] + (n,)
        output = output.reshape(output_shape)

    return output


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


