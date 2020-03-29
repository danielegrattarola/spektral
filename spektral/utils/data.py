import numpy as np
import scipy.sparse as sp


def numpy_to_disjoint(X_list, A_list, E_list=None):
    """
    Converts a batch of graphs stored in lists (X, A, and optionally E) to the
    [disjoint mode](https://danielegrattarola.github.io/spektral/data/).
    Note that the number of nodes N can vary between graphs, although each entry
    i of each list should be associated to the same graph, i.e.,
    `X_list[i].shape[0] == A_list[i].shape[0] == E_list[i].shape[0]`.

    :param X_list: a list of np.arrays of shape (N, F);
    :param A_list: a list of np.arrays or sparse matrices of shape (N, N);
    :param E_list: a list of np.arrays of shape (N, N, S);
    :return:
    """
    X_out = np.vstack(X_list)
    A_list = [sp.coo_matrix(a) for a in A_list]
    if E_list is not None:
        if E_list[0].ndim == 3:
            E_list = [e[a.row, a.col] for e, a in zip(E_list, A_list)]
        E_out = np.vstack(E_list)
    A_out = sp.block_diag(A_list)
    n_nodes = np.array([x.shape[0] for x in X_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    if E_list is not None:
        return X_out, A_out, E_out, I_out
    else:
        return X_out, A_out, I_out


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