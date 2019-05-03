import numpy as np
import scipy.sparse as sp


class Batch(object):
    """
    Wraps a batch of graphs.
    """
    def __init__(self, A_list, X_list, E_list=None):
        self.A_list = A_list
        self.X_list = X_list
        self.E_list = E_list

        n_nodes = np.array([a_.shape[0] for a_ in self.A_list])
        self.S_list = np.repeat(np.arange(len(n_nodes)), n_nodes)

    def get(self):
        A_out = sp.block_diag(self.A_list)
        X_out = np.vstack(self.X_list)

        return A_out, X_out, self.S_list
