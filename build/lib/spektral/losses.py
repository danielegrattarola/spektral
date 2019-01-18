from __future__ import absolute_import

import tensorflow as tf
from keras import backend as K

from .layers import MaxPoolMatching, Affinity


def _check_loss_weights(loss_weights):
    if loss_weights is not None:
        assert len(loss_weights) == 3, 'When passing custom loss ' \
                                       'multiplicators, it is required to ' \
                                       'specify a weight for each of the ' \
                                       'three terms of the loss (even if ' \
                                       'it will not be used).'
        return [float(t) for t in loss_weights]
    else:
        return [1., 1., 1.]


def graph_loss(adj_in, adj_out, nf_in=None, nf_out=None, ef_in=None,
               ef_out=None, loss_weights=None, sum_output=False,
               pw_adj=None, pw_nf=None, pw_ef=None):
    """
    Computes reconstruction loss for a graph autoencoder, as defined in 
    Simonovsky and Komodakis, 2018.
    :param adj_in: input adjacency matrices
    :param adj_out: reconstructed adjacency matrices
    :param nf_in: input node features matrices
    :param nf_out: reconstructed node features matrices
    :param ef_in: input edge feature matrices
    :param ef_out: reconstructed edge feature matrices
    :param loss_weights: list of 3 floats with which to weight each loss term
    :param sum_output: sum the output tensors and return a single tensor
    :param pw_adj: positive weights with which to re-weight the loss for adj
    :param pw_nf: positive weights with which to re-weight the loss for nf
    :param pw_ef: positive weights with which to re-weight the loss for ef
    :return: loss tensor
    """
    loss_weights = _check_loss_weights(loss_weights)

    N = K.int_shape(adj_in)[-1]  # Max number of nodes
    true_N = K.sum(tf.matrix_diag_part(adj_in), -1)  # True number of nodes in each sample
    node_fltr = K.eye(N)
    edge_fltr = (K.ones((N, N)) - node_fltr)
    adj_loss = K.binary_crossentropy(adj_in, adj_out)
    if pw_adj is not None:
        sample_weights = 1 + pw_adj * adj_in
        adj_loss *= sample_weights
    adj_loss_n = K.sum(node_fltr * adj_loss, (-1, -2)) / N
    adj_loss_e = K.sum(edge_fltr * adj_loss, (-1, -2)) / (N * (N - 1))
    adj_loss = K.mean(adj_loss_n + adj_loss_e)
    adj_loss *= loss_weights[0]

    # Compute node features loss
    if nf_in is not None:
        nf_loss = K.categorical_crossentropy(nf_in, nf_out)
        if pw_nf is not None:
            sample_weights = 1 + K.sum(pw_nf * nf_in, -1)
            nf_loss *= sample_weights
        nf_loss /= N
        nf_loss = K.mean(nf_loss)
        nf_loss *= loss_weights[1]
    else:
        nf_loss = 0.

    # Compute edge features loss
    if ef_in is not None:
        ef_loss = K.categorical_crossentropy(ef_in, ef_out)
        if pw_ef is not None:
            sample_weights = 1 + K.sum(pw_ef * ef_in, -1)
            ef_loss *= sample_weights
        ef_loss = K.sum(ef_loss, (-1, -2))
        ef_normalization = K.sum(adj_in, (-1, -2)) - true_N
        ef_loss /= ef_normalization
        ef_loss = K.mean(ef_loss)
        ef_loss *= loss_weights[2]
    else:
        ef_loss = 0.

    if sum_output:
        return adj_loss + nf_loss + ef_loss
    else:
        return adj_loss, nf_loss, ef_loss


def graph_loss_mpm(affinity_function, adj_in, adj_out, nf_in=None,
                   nf_out=None, ef_in=None, ef_out=None, loss_weights=None,
                   iters=75, sum_output=False, pw_adj=None, pw_nf=None,
                   pw_ef=None):
    """
    Computes reconstruction loss for a graph autoencoder, as defined in 
    Simonovsky and Komodakis, 2018, with an additional graph matching step to 
    account for possible node permutations.
    :param affinity_function: affinity function to compute graph matching (see 
        layers.base.Affinity)
    :param adj_in: input adjacency matrices
    :param adj_out: reconstructed adjacency matrices
    :param nf_in: input node features matrices
    :param nf_out: reconstructed node features matrices
    :param ef_in: input edge feature matrices
    :param ef_out: reconstructed edge feature matrices
    :param loss_weights: list of 3 floats with which to weight each loss term
    :param iters: number of graph matching iterations
    :param sum_output: sum the output tensors and return a single tensor
    :param pw_adj: positive weights with which to re-weight the loss for adj
    :param pw_nf: positive weights with which to re-weight the loss for nf
    :param pw_ef: positive weights with which to re-weight the loss for ef
    :return: loss tensor
    """
    loss_weights = _check_loss_weights(loss_weights)

    N = K.int_shape(adj_in)[-1]
    true_N = K.sum(tf.matrix_diag_part(adj_in), -1)  # True number of nodes in each sample
    F = K.int_shape(nf_in)[-1] if nf_in is not None else None
    S = K.int_shape(ef_in)[-1] if ef_in is not None else None
    node_fltr = K.eye(N)
    edge_fltr = (K.ones((N, N)) - node_fltr)

    # Match graphs based on affinity function
    aff_args = [i for i in [adj_in, adj_out, nf_in, nf_out, ef_in, ef_out] if i is not None]
    affinity = Affinity(affinity_function, N, F, S)(aff_args)
    matching = MaxPoolMatching(iters=iters)([affinity, adj_in, adj_out])
    matching_T = K.permute_dimensions(matching, (0, 2, 1))

    # Match adjacency matrices and compute loss
    adj_in_match = K.batch_dot(K.batch_dot(matching, adj_in), matching_T)
    adj_loss = K.binary_crossentropy(adj_in_match, adj_out)

    # Compute and match sample weights
    if pw_adj is not None:
        sw = 1 + pw_adj * adj_in
        sw_match = K.batch_dot(K.batch_dot(matching, sw), matching_T)
        adj_loss *= sw_match

    adj_loss_n = K.sum(node_fltr * adj_loss, (-1, -2)) / N
    adj_loss_e = K.sum(edge_fltr * adj_loss, (-1, -2)) / (N * (N - 1))
    adj_loss = K.mean(adj_loss_n + adj_loss_e)
    adj_loss *= loss_weights[0]

    # Match node features and compute loss
    if nf_in is not None:
        nf_out_match = K.batch_dot(matching_T, nf_out)
        nf_loss = K.categorical_crossentropy(nf_in, nf_out_match)

        # Compute and match sample weights
        if pw_nf is not None:
            sw = 1 + K.sum(pw_nf * nf_in, -1)
            sw_match = K.batch_dot(matching_T, sw)
            nf_loss *= sw_match

        nf_loss /= N
        nf_loss = K.mean(nf_loss)
        nf_loss *= loss_weights[1]
    else:
        nf_loss = 0.

    # Match edge features and compute loss
    if ef_in is not None:
        ef_out_match = [K.batch_dot(matching_T, ef_out[..., s]) for s in range(S)]
        ef_out_match = [K.batch_dot(e_o_m, matching) for e_o_m in ef_out_match]
        ef_out_match = K.stack(ef_out_match, -1)
        ef_loss = K.categorical_crossentropy(ef_in, ef_out_match)

        # Compute and match sample weights
        if pw_ef is not None:
            sw = 1 + K.sum(pw_ef * ef_in, -1)
            sw_match = K.batch_dot(matching_T, sw)
            ef_loss *= sw_match

        ef_loss = K.sum(ef_loss, (-1, -2))
        ef_normalization = K.sum(adj_in_match, (-1, -2)) - true_N
        ef_loss /= ef_normalization
        ef_loss = K.mean(ef_loss)
        ef_loss *= loss_weights[2]
    else:
        ef_loss = 0.

    if sum_output:
        return adj_loss + nf_loss + ef_loss
    else:
        return adj_loss, nf_loss, ef_loss


def graph_variational_loss(adj_in, adj_out, z_mean, z_log_var, nf_in=None,
                           nf_out=None, ef_in=None, ef_out=None,
                           loss_weights=None, sum_output=False, pw_adj=None,
                           pw_nf=None, pw_ef=None):
    """
    Computes reconstruction loss for a graph variationa autoencoder, as defined 
    in Simonovsky and Komodakis, 2018.
    :param adj_in: input adjacency matrices
    :param adj_out: reconstructed adjacency matrices
    :param z_mean: tensor of means produced by the encoder
    :param z_log_var: tensor of log standard deviations produced by the encoder
    :param nf_in: input node features matrices
    :param nf_out: reconstructed node features matrices
    :param ef_in: input edge feature matrices
    :param ef_out: reconstructed edge feature matrices
    :param loss_weights: list of 3 floats with which to weight each loss term
    :param sum_output: sum the output tensors and return a single tensor
    :param pw_adj: positive weights with which to re-weight the loss for adj
    :param pw_nf: positive weights with which to re-weight the loss for nf
    :param pw_ef: positive weights with which to re-weight the loss for ef
    :return: loss tensor
    """
    adj_loss, nf_loss, ef_loss = graph_loss(adj_in, adj_out,
                                            nf_in=nf_in, nf_out=nf_out,
                                            ef_in=ef_in, ef_out=ef_out,
                                            loss_weights=loss_weights,
                                            sum_output=False,
                                            pw_adj=pw_adj,
                                            pw_nf=pw_nf,
                                            pw_ef=pw_ef)

    # Kullback-Leibler regularization term
    kl_loss = (- 0.5) * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = K.mean(kl_loss)

    if sum_output:
        return adj_loss + nf_loss + ef_loss + kl_loss
    else:
        return adj_loss, nf_loss, ef_loss, kl_loss


def graph_variational_loss_mpm(affinity_function, adj_in, adj_out, z_mean,
                               z_log_var, nf_in=None, nf_out=None, ef_in=None,
                               ef_out=None, loss_weights=None, iters=75,
                               sum_output=False, pw_adj=None, pw_nf=None,
                               pw_ef=None):
    """
    Computes reconstruction loss for a graph variationa autoencoder, as defined 
    in Simonovsky and Komodakis, 2018, with an additional graph matching step to 
    account for possible node permutations.
    :param affinity_function: affinity function to compute graph matching (see 
        layers.base.Affinity)
    :param adj_in: input adjacency matrices
    :param adj_out: reconstructed adjacency matrices
    :param z_mean: tensor of means produced by the encoder
    :param z_log_var: tensor of log standard deviations produced by the encoder
    :param nf_in: input node features matrices
    :param nf_out: reconstructed node features matrices
    :param ef_in: input edge feature matrices
    :param ef_out: reconstructed edge feature matrices
    :param loss_weights: list of 3 floats with which to weight each loss term
    :param iters: number of graph matching iterations
    :param sum_output: sum the output tensors and return a single tensor
    :param pw_adj: positive weights with which to re-weight the loss for adj
    :param pw_nf: positive weights with which to re-weight the loss for nf
    :param pw_ef: positive weights with which to re-weight the loss for ef
    :return: loss tensor
    """
    adj_loss, nf_loss, ef_loss = graph_loss_mpm(affinity_function,
                                                adj_in, adj_out,
                                                nf_in=nf_in, nf_out=nf_out,
                                                ef_in=ef_in, ef_out=ef_out,
                                                loss_weights=loss_weights,
                                                iters=iters,
                                                sum_output=False,
                                                pw_adj=pw_adj,
                                                pw_nf=pw_nf,
                                                pw_ef=pw_ef)
    # Kullback-Leibler regularization term
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = K.mean(kl_loss)

    if sum_output:
        return adj_loss + kl_loss + nf_loss + ef_loss
    else:
        return adj_loss, kl_loss, nf_loss, ef_loss