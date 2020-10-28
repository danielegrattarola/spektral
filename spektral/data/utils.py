import numpy as np
import tensorflow as tf
from scipy import sparse as sp

from spektral.utils import pad_jagged_array


def numpy_to_disjoint(x_list, a_list, e_list=None):
    """
    Converts lists of node features, adjacency matrices and (optionally) edge
    features to [disjoint mode](https://danielegrattarola.github.io/spektral/data/#disjoint-mode).

    The i-th element of each list must be associated with the i-th graph.

    The method also computes the batch index to retrieve individual graphs
    from the disjoint union.

    :param x_list: a list of np.arrays of shape `(N, F)` -- note that `N` can
    change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(N, N)`;
    :param e_list: a list of np.arrays of shape `(N, N, S)`;
    :return:
        -  `x`: np.array of shape `(n_nodes, F)`;
        -  `a`: scipy.sparse matrix of shape `(n_nodes, n_nodes)`;
        -  `e`: (only if `e_list` is given) np.array of shape `(n_edges, S)`;
        -  `i`: np.array of shape `(n_nodes, )`;
    """
    x_out = np.vstack(x_list)
    a_list = [sp.coo_matrix(a) for a in a_list]
    if e_list is not None:
        if e_list[0].ndim == 3:
            e_list = [e[a.row, a.col] for e, a in zip(e_list, a_list)]
        e_out = np.vstack(e_list)
    a_out = sp.block_diag(a_list)
    n_nodes = np.array([x.shape[0] for x in x_list])
    i_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    if e_list is not None:
        return x_out, a_out, e_out, i_out
    else:
        return x_out, a_out, i_out


def numpy_to_batch(x_list, a_list, e_list=None):
    """
    Converts lists of node features, adjacency matrices and (optionally) edge 
    features to [batch mode](https://danielegrattarola.github.io/spektral/data/#batch-mode),
    by zero-padding all X, A and E matrices to have the same node dimensions.

    The i-th element of each list must be associated with the i-th graph.

    Note that if `a_list` contains sparse matrices, they will be converted to
    dense np.arrays, which can be memory-expensive.

    :param x_list: a list of np.arrays of shape `(N, F)` -- note that `N` can
    change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(N, N)`;
    :param e_list: a list of np.arrays of shape `(N, N, S)`;
    :return:
        -  `x`: np.array of shape `(batch, n_max, F)`;
        -  `a`: np.array of shape `(batch, n_max, n_max)`;
        -  `e`: (only if `e_list` is given) np.array of shape
        `(batch, n_max, n_max, S)`;
    """
    n_max = max([a.shape[-1] for a in a_list])
    x_out = pad_jagged_array(x_list, (n_max, -1))
    # Convert sparse matrices to dense
    if hasattr(a_list[0], 'toarray'):
        a_list = [a.toarray() for a in a_list]
    a_out = pad_jagged_array(a_list, (n_max, n_max))
    if e_list is not None:
        e_out = pad_jagged_array(e_list, (n_max, n_max, -1))
        return x_out, a_out, e_out
    else:
        return x_out, a_out


def batch_generator(data, batch_size=32, epochs=1, shuffle=True):
    """
    Iterates over the data for the given number of epochs, yielding batches of
    size `batch_size`.
    :param data: np.array or list of np.arrays with the same first dimension;
    :param batch_size: number of samples in a batch;
    :param epochs: number of times to iterate over the data;
    :param shuffle: whether to shuffle the data at the beginning of each epoch
    :return: batches of size `batch_size`.
    """
    if not isinstance(data, list):
        data = [data]
    if len(data) < 1:
        raise ValueError('data cannot be empty')
    if len(set([len(item) for item in data])) > 1:
        raise ValueError('All inputs must have the same __len__')

    len_data = len(data[0])
    batches_per_epoch = int(np.ceil(len_data / batch_size))
    for epochs in range(epochs):
        if shuffle:
            shuffle_inplace(*data)
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)

            to_yield = [item[start:stop] for item in data]
            if len(data) == 1:
                to_yield = to_yield[0]

            yield to_yield


def shuffle_inplace(*args):
    rng_state = np.random.get_state()
    for a in args:
        np.random.set_state(rng_state)
        np.random.shuffle(a)


def get_spec(x):
    if isinstance(x, tf.SparseTensor) or sp.issparse(x):
        return tf.SparseTensorSpec
    else:
        return tf.TensorSpec


def prepend_none(t):
    return (None, ) + t


def output_signature(signature):
    output = []
    keys = ['x', 'a', 'e', 'i']
    for k in keys:
        if k in signature:
            shape = signature[k]['shape']
            dtype = signature[k]['dtype']
            spec = signature[k]['spec']
            output.append(spec(shape, dtype))
    output = tuple(output)
    if 'y' in signature:
        shape = signature['y']['shape']
        dtype = signature['y']['dtype']
        spec = signature['y']['spec']
        output = (output, spec(shape, dtype))

    return output
