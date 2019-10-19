This module provides some functions to create functional connectivity
networks, and requires the [dyfunconn](https://dyfunconn.readthedocs.io/)
library to be installed on the system.

### get_fc_graphs


```python
spektral.brain.get_fc_graphs(x, fc_measure, nf_mode, samples_per_graph=None, link_cutoff=0.0, percentiles=None, self_loops=True, band_freq=None, band_freq_hi=None, sampling_freq=None, nfft=None, n_overlap=None)
```



Compute a sequence of functional connectivity networks from the given time
series. Each graph will be computed using consecutive windows of
`samples_per_graph` timesteps. If x.shape[1] % samples_per_graph != 0, then
the remaining time steps at the end of the time series are discarded.

**Arguments**  

- ` x`: np.array of shape (n_channels, n_timesteps)

- ` fc_measure`: string, the FC measure to use (see _get_fc_graph())

- ` nf_mode`: string, how to compute node features (see _get_fc_graph())

- ` samples_per_graph`: int, number of samples used to compute a graph.

- ` link_cutoff`: remove links in the FC graphs if the absolute value of
their FC measure is smaller than the threshold. Overrides `percentiles` flag.

- ` percentiles`: tuple of 2 floats, keep only links whose FC measure is
below the `percentiles[0]`-th percentile or above the `percentiles[1]`-th
percentile.

- ` self_loops`: add self-loops to the adjacency matrices.

- ` band_freq`: tuple of 2 floats >0, filter the signal with a bandpass
filter in this band. mandatory when using dyfunconn.

- ` band_freq_hi`: mandatory when using dyfunconn.aec().

- ` sampling_freq`: sampling frequency of the data. Mandatory when
using dyfunconn.

- ` nfft`: mandatory when using dyfunconn.wpli() and dyfunconn.dwpli().

- ` n_overlap`: mandatory when using dyfunconn.wpli() and dyfunconn.dwpli().

**Return**  

