from __future__ import absolute_import

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence


def degree(adj):
    """
    Computes the degree matrix of the given adjacency matrix.
    :param adj: rank 2 array or sparse matrix
    :return: the degree matrix in sparse DIA format
    """
    degrees = np.array(adj.sum(1)).flatten()
    return sp.diags(degrees)


def degree_power(adj, pow):
    """
    Computes \(D^{p}\) from the given adjacency matrix. Useful for computing
    normalised Laplacians.
    :param adj: rank 2 array or sparse matrix
    :param pow: exponent to which elevate the degree matrix
    :return: the exponentiated degree matrix in sparse DIA format
    """
    degrees = np.power(np.array(adj.sum(1)), pow).flatten()
    degrees[np.isinf(degrees)] = 0.
    return sp.diags(degrees, 0)


def normalized_adjacency(adj, symmetric=True):
    """
    Normalizes the given adjacency matrix using the degree matrix as either
    \(D^{-1}A\) or \(D^{-1/2}AD^{-1/2}\) (symmetric normalization).
    :param adj: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(adj, -0.5)
        if sp.issparse(adj):
            output = normalized_D.dot(adj).dot(normalized_D)
        else:
            normalized_D = normalized_D.toarray()
            output = normalized_D.dot(adj).dot(normalized_D)
    else:
        normalized_D = degree_power(adj, -1.)
        output = normalized_D.dot(adj)

    return output


def laplacian(adj):
    """
    Computes the Laplacian of the given adjacency matrix as \(D - A\).
    :param adj: rank 2 array or sparse matrix;
    :return: the Laplacian.
    """
    return degree(adj) - adj


def normalized_laplacian(adj, symmetric=True):
    """
    Computes a  normalized Laplacian of the given adjacency matrix as
    \(I - D^{-1}A\) or \(I - D^{-1/2}AD^{-1/2}\) (symmetric normalization).
    :param adj: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized Laplacian.
    """
    I = sp.eye(adj.shape[-1], dtype=adj.dtype)
    normalized_adj = normalized_adjacency(adj, symmetric=symmetric)
    return I - normalized_adj


def rescale_laplacian(L, lmax=None):
    """
    Rescales the Laplacian eigenvalues in [-1,1], using lmax as largest eigenvalue.
    """
    if lmax is None:
        try:
            lmax = sp.linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
        except ArpackNoConvergence:
            lmax = 2
    L_scaled = (2. / lmax) * L - sp.eye(L.shape[0], dtype=L.dtype)
    return L_scaled


def localpooling_filter(adj, symmetric=True):
    """
    Computes the local pooling filter from the given adjacency matrix, as 
    described by Kipf & Welling (2017).
    :param adj: a np.array or scipy.sparse matrix of rank 2 or 3;
    :param symmetric: boolean, whether to normalize the matrix as
    \(D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}\) or as \(D^{-1}A\);
    :return: the filter matrix, as dense np.array.
    """
    fltr = adj.copy()
    I = sp.eye(adj.shape[-1], dtype=adj.dtype)
    if adj.ndim == 3:
        for i in range(adj.shape[0]):
            A_tilde = adj[i] + I
            fltr[i] = normalized_adjacency(A_tilde, symmetric=symmetric)
    else:
        A_tilde = adj + I
        fltr = normalized_adjacency(A_tilde, symmetric=symmetric)
    return fltr


def chebyshev_polynomial(X, k):
    """
    Calculates Chebyshev polynomials up to order k.
    :param X: a np.array or scipy.sparse matrix;
    :param k: the order up to which compute the polynomials,
    :return: a list of k + 1 sparse matrices with one element for each degree of 
            the approximation.
    """
    T_k = list()
    T_k.append(sp.eye(X.shape[0], dtype=X.dtype).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def chebyshev_filter(adj, k, symmetric=True):
    """
    Computes the Chebyshev filter from the given adjacency matrix, as described
    in Defferrard et at. (2016).
    :param adj: a np.array or scipy.sparse matrix;
    :param k: integer, the order up to which to compute the Chebyshev polynomials;
    :param symmetric: boolean, whether to normalize the matrix as
    \(D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}\) or as \(D^{-1}A\);
    :return: a list of k+1 filter matrices, as np.arrays.
    """
    normalized_adj = normalized_adjacency(adj, symmetric)
    L = sp.eye(adj.shape[0], dtype=adj.dtype) - normalized_adj  # Compute Laplacian

    # Rescale Laplacian
    L_scaled = rescale_laplacian(L)

    # Compute Chebyshev polynomial approximation
    T_k = chebyshev_polynomial(L_scaled, k)

    return T_k


