import numpy as np
from scipy import stats

try:
    from dyfunconn import fc
except ImportError:
    fc = None


def get_fc_graphs(x, fc_measure, nf_mode,
                  samples_per_graph=None, link_cutoff=0., percentiles=None,
                  self_loops=True, band_freq=None, band_freq_hi=None,
                  sampling_freq=None, nfft=None, n_overlap=None):
    """
    Compute a sequence of functional connectivity networks from the given time
    series. Each graph will be computed using consecutive windows of
    `samples_per_graph` timesteps. If x.shape[1] % samples_per_graph != 0, then
    the remaining time steps at the end of the time series are discarded.
    :param x: np.array of shape (n_channels, n_timesteps)
    :param fc_measure: string, the FC measure to use (see _get_fc_graph())
    :param nf_mode: string, how to compute node features (see _get_fc_graph())
    :param samples_per_graph: int, number of samples used to compute a graph.
    :param link_cutoff: remove links in the FC graphs if the absolute value of
    their FC measure is smaller than the threshold. Overrides `percentiles` flag.
    :param percentiles: tuple of 2 floats, keep only links whose FC measure is
    below the `percentiles[0]`-th percentile or above the `percentiles[1]`-th
    percentile.
    :param self_loops: add self-loops to the adjacency matrices.
    :param band_freq: tuple of 2 floats >0, filter the signal with a bandpass
    filter in this band. mandatory when using dyfunconn.
    :param band_freq_hi: mandatory when using dyfunconn.aec().
    :param sampling_freq: sampling frequency of the data. Mandatory when
    using dyfunconn.
    :param nfft: mandatory when using dyfunconn.wpli() and dyfunconn.dwpli().
    :param n_overlap: mandatory when using dyfunconn.wpli() and dyfunconn.dwpli().
    :return:
    """
    if fc is None:
        raise ImportError('Creating functional connectivity networks '
                          'requires dyfunconn.')
    if x.ndim != 2:
        raise ValueError('Expected x to have rank 2, got {}'.format(x.ndim))
    if samples_per_graph is None:
        samples_per_graph = x.shape[1]

    output = _loop(
        x, samples_per_graph, fc_measure, nf_mode,
        band_freq=band_freq, band_freq_hi=band_freq_hi, sampling_freq=sampling_freq,
        nfft=nfft, n_overlap=n_overlap
    )

    nf, ef = zip(*output)
    nf = np.array(nf)
    ef = np.array(ef)

    # Compute adjacency matrix
    adj = np.ones_like(ef)
    if link_cutoff != 0:
        adj[np.abs(adj) < link_cutoff] = 0
    elif percentiles is not None:
        N = adj.shape[-1]
        pctl_lo = []
        pctl_hi = []
        for fcm in ef.reshape((-1, N * N)).T:
            pctl_lo.append(stats.scoreatpercentile(fcm, percentiles[0]))
            pctl_hi.append(stats.scoreatpercentile(fcm, percentiles[1]))
        pctl_lo = np.array(pctl_lo).reshape((-1, N, N))
        pctl_hi = np.array(pctl_hi).reshape((-1, N, N))
        adj = np.logical_or(ef < pctl_lo, ef > pctl_hi).astype(np.int)
    else:
        pass

    if self_loops:
        for i in range(adj.shape[0]):
            np.fill_diagonal(adj[i], 1.0)
    else:
        # Set the main diagonal to zero
        for i in range(adj.shape[0]):
            np.fill_diagonal(adj[i], 0.0)
            np.fill_diagonal(ef[i], 0.0)

    nf = nf[..., None]
    ef = ef[..., None]

    return adj, nf, ef


def _loop(x, samples_per_graph, fc_measure='corr', nf_mode='mean',
          band_freq=None, band_freq_hi=None, sampling_freq=None, nfft=None,
          n_overlap=None):
    output = [
        _get_fc_graph(
            x[:, i: i + samples_per_graph], fc_measure, nf_mode,
            band_freq=band_freq, band_freq_hi=band_freq_hi,
            sampling_freq=sampling_freq, nfft=nfft, n_overlap=n_overlap)
        for i in range(0, x.shape[1], samples_per_graph)
    ]
    return output


def _get_fc_graph(x, fc_measure, nf_mode, band_freq=None, band_freq_hi=None,
                  sampling_freq=None, nfft=None, n_overlap=None):
    if fc_measure == 'iplv':
        _, ef = fc.iplv(x, band_freq, sampling_freq)
    elif fc_measure == 'icoh':
        ef = fc.icoherence(x, band_freq, sampling_freq)
    elif fc_measure == 'corr':
        ef = fc.corr(x, band_freq, sampling_freq)
    elif fc_measure == 'aec':
        ef = fc.aec(x, band_freq, band_freq_hi, sampling_freq)
    elif fc_measure == 'wpli':
        csdparams = {'NFFT': nfft, 'noverlap': n_overlap}
        ef = fc.wpli(x, band_freq, sampling_freq, **csdparams)
    elif fc_measure == 'dwpli':
        csdparams = {'NFFT': nfft, 'noverlap': n_overlap}
        ef = fc.dwpli(x, band_freq, sampling_freq, **csdparams)
    elif fc_measure == 'dpli':
        ef = fc.dpli(x, band_freq, sampling_freq)
    else:
        raise ValueError('Invalid fc_measure ' + fc_measure)

    # Dummy node features
    if nf_mode == 'full':
        nf = x.copy()
    elif nf_mode == 'mean':
        nf = np.mean(x, -1)
    elif nf_mode == 'energy':
        nf = np.sum(x ** 2, -1)
    elif nf_mode == 'power':
        nf = np.sum(x ** 2, -1) / x.shape[-1]
    elif nf_mode == 'ones':
        nf = np.ones((ef.shape[0], 1))
    else:
        raise ValueError('Invalid nf_mode' + nf_mode)
    return nf, ef
