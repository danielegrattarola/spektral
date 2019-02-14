import numpy as np
from scipy import stats
from joblib import Parallel, delayed
try:
    from dyfunconn import fc
except ImportError:
    fc = None


def _get_fc_graph(x, band_freq, sampling_freq, fc_measure='corr',
                  link_cutoff=0., band_freq_hi=(20., 45.), nfft=128,
                  n_overlap=64):
    """
    Build a functional connectivity network from the given data stream.
    :param x: numpy array of shape (n_channels, n_samples);
    :param band_freq: list with two elements, the band in which to estimate FC;
    :param sampling_freq: float, sampling frequency of the stream;
    :param fc_measure: functional connectivity measure to use;
    :param link_cutoff: links with absolute FC measure below this value will be
      removed;
    :param band_freq_hi: high band used to estimate FC when using 'aec';
    :param nfft: TODO, affects 'wpli' and 'dwpli';
    :param n_overlap: TODO, affects 'wpli' and 'dwpli';
    :return: FC graph in numpy format (note that node features are all ones).
    """
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
        ef = np.abs(ef)
    elif fc_measure == 'dwpli':
        csdparams = {'NFFT': nfft, 'noverlap': n_overlap}
        ef = fc.dwpli(x, band_freq, sampling_freq, **csdparams)
        ef = np.abs(ef)
    elif fc_measure == 'dpli':
        ef = fc.dpli(x, band_freq, sampling_freq)
    else:
        raise ValueError('Invalid fc_measure')

    # Set the main diagonal to zero (no self-loops)
    np.fill_diagonal(ef, 0.0)

    # Compute adjacency matrix by rounding to 0 and 1 based on the cutoff
    adj = ef.copy()
    if link_cutoff != 0:
        adj[np.abs(adj) >= link_cutoff] = 1
        adj[np.abs(adj) < link_cutoff] = 0
    else:
        adj[...] = 1

    # Dummy node features
    # TODO: proper nf
    nf = np.ones((ef.shape[0], 1))

    # Edge features
    ef = ef[..., None]

    return adj, nf, ef


def get_fc(x, band_freq, sampling_freq, samples_per_graph=None,
           fc_measure='corr', link_cutoff=0., percentiles=None,
           band_freq_hi=(20., 45.), nfft=128, n_overlap=64, njobs=1):
    """
    Build functional connectivity networks from the given data stream.
    :param x: numpy array of shape (n_channels, n_samples);
    :param band_freq: list with two elements, the band in which to estimate FC;
    :param sampling_freq: float, sampling frequency of the stream;
    :param samples_per_graph: number of samples to use to generate a graph. By 
    default, the whole stream is used. If provided, 
    1 + (n_samples / samples_per_graph) will be generated;
    :param fc_measure: functional connectivity measure to use;
    :param link_cutoff: links with absolute FC measure below this value will be
      removed;
    :param percentiles: tuple of two numbers >0 and <100; links with FC measure 
    between the two percentiles will be removed (statistics are calculated for
    each edge). Note that this option ignores `link_cutoff`.
    :param band_freq_hi: high band used to estimate FC when using 'aec';
    :param nfft: TODO, affects 'wpli' and 'dwpli';
    :param n_overlap: TODO, affects 'wpli' and 'dwpli';
    :param njobs: number of processes to use (-1 to use all available cores);
    :return: FC graph(s) in numpy format (note that node features are all ones).
    """
    if fc is None:
        raise ImportError('`get_fc` requires dyfunconn.')
    if x.ndim != 2:
        raise ValueError('Expected x to have rank 2, got {}'.format(x.ndim))
    if samples_per_graph is None:
        samples_per_graph = x.shape[1]

    output = Parallel(n_jobs=njobs)(
        delayed(_get_fc_graph)(x[:, i: i + samples_per_graph],
                               band_freq,
                               sampling_freq,
                               fc_measure=fc_measure,
                               link_cutoff=link_cutoff,
                               band_freq_hi=band_freq_hi,
                               nfft=nfft,
                               n_overlap=n_overlap)
        for i in range(0, x.shape[1], samples_per_graph)
    )

    adj, nf, ef = zip(*output)
    adj = np.array(adj)
    nf = np.array(nf)
    ef = np.array(ef)

    if percentiles is not None:
        N = adj.shape[-1]
        pctl_lo = []
        pctl_hi = []
        for fcm in ef.reshape((-1, N * N)).T:
            pctl_lo.append(stats.scoreatpercentile(fcm, percentiles[0]))
            pctl_hi.append(stats.scoreatpercentile(fcm, percentiles[1]))
        pctl_lo = np.array(pctl_lo).reshape((1, N, N, 1))
        pctl_hi = np.array(pctl_hi).reshape((1, N, N, 1))
        adj = np.logical_or(ef < pctl_lo, ef > pctl_hi).astype(np.int)[..., 0]

    return adj, nf, ef
