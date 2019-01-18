from __future__ import absolute_import

from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence

from .misc import add_eye, normalize_adj


def localpooling_filter(adj, symmetric_normalization=True):
    """
    Computes the local pooling filter from the given adjacency matrix, as 
    described by Kipf & Welling (2017).
    :param adj: a np.array or scipy.sparse matrix of rank 2 or 3
    :param symmetric_normalization: boolean, whether to normalize the matrix as 
    \(D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}\) or as \(D^{-1}A\).
    :return: the filter matrix, as dense np.array
    """
    if adj.ndim == 3:
        for i in range(adj.shape[0]):
            adj[i] = add_eye(adj[i])
            adj[i] = normalize_adj(adj[i], symmetric_normalization)
    else:
        adj = add_eye(adj)
        adj = normalize_adj(adj, symmetric_normalization)

    return adj


def chebyshev_polynomial(X, k):
    """
    Calculates Chebyshev polynomials up to order k.
    :param X: a np.array or scipy.sparse matrix
    :param k: the order up to which compute the polynomials
    :return: a list of k + 1 sparse matrices with one element for each degree of 
            the approximation
    """
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def chebyshev_filter(adj, k, symmetric_normalization=True):
    """
    Computes the Chebyshev filter from the given adjacency matrix, as described
    in Defferrard et at. (2016).
    :param adj: a np.array or scipy.sparse matrix
    :param k: integer, the order up to which to cocmpute the Chebyshev polynomials
    :param symmetric_normalization: boolean, whether to normalize the matrix as 
    \(D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}\) or as \(D^{-1}A\). 
    :return: a list of k+1 filter matrices, as np.arrays 
    """
    adj_normalized = normalize_adj(adj, symmetric_normalization)
    L = sp.eye(adj.shape[0]) - adj_normalized  # Compute Laplacian

    # Rescale Laplacian
    try:
        largest_eigval = sp.linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        largest_eigval = 2
    L_scaled = (2. / largest_eigval) * L - sp.eye(L.shape[0])

    # Compute Chebyshev polynomial approximation
    T_k = chebyshev_polynomial(L_scaled, k)

    return T_k