from keras import backend as K
from keras.backend import tf
import scipy.sparse as sp
import numpy as np


def sparse_bool_mask(x, mask, axis=0):
    # Only necessary if indices may have non-unique elements
    indices = tf.boolean_mask(tf.range(tf.shape(x)[axis]), mask)
    n_indices = tf.size(indices)
    # Get indices for the axis
    idx = x.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))  # TODO this has quadratic cost
    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(x.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tf.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(x.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis],
                             tf.expand_dims(idx_new, 1),
                             indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([x.dense_shape[:axis],
                           [n_indices],
                           x.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)


def mixed_mode_dot(fltr, features):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', fltr, output)`, but
    works for both dense and sparse input filters.
    :param fltr: rank 2 tensor, the filter for convolution
    :param features: rank 3 tensor, the features of the input signals
    :return:
    """
    _, m_, f_ = K.int_shape(features)
    features = K.permute_dimensions(features, [1, 2, 0])
    features = K.reshape(features, (m_, -1))
    features = K.dot(fltr, features)
    features = K.reshape(features, (m_, f_, -1))
    features = K.permute_dimensions(features, [2, 0, 1])

    return features


def filter_dot(fltr, features):
    """
    Performs the multiplication of a graph filter (N x N) with the node features,
    automatically dealing with single, mixed, and batch modes.
    :param fltr: the graph filter(s) (N x N in single and mixed mode,
    batch x N x N in batch mode).
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)


def sp_matrix_to_sp_tensor_value(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensorValue
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensorValue
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return K.tf.SparseTensorValue(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensor
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensor
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return K.tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def matrix_power(x, k):
    """
    Computes the k-th power of a square matrix.
    :param x: a square matrix (Tensor or SparseTensor)
    :param k: exponent
    :return: matrix of same type and dtype as the input
    """
    if K.is_sparse(x):
        sparse = True
        x_dense = tf.sparse.to_dense(x)
    else:
        sparse = False
        x_dense = x

    x_k = x_dense
    for _ in range(k - 1):
        x_k = K.dot(x_k, x_dense)

    if sparse:
        return tf.contrib.layers.dense_to_sparse(x_k)
    else:
        return x_k