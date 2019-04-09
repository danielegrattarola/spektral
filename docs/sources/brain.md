This module provides some functions to create functional connectivity
networks, and requires the [dyfunconn](https://dyfunconn.readthedocs.io/)
library to be installed on the system.

### get_fc


```python
spektral.brain.get_fc(x, band_freq, sampling_freq, samples_per_graph=None, fc_measure='corr', link_cutoff=0.0, percentiles=None, band_freq_hi=(20.0, 45.0), nfft=128, n_overlap=64, njobs=1)
```



Build functional connectivity networks from the given data stream.

**Arguments**  

- ` x`: numpy array of shape (n_channels, n_samples);

- ` band_freq`: list with two elements, the band in which to estimate FC;

- ` sampling_freq`: float, sampling frequency of the stream;

- ` samples_per_graph`: number of samples to use to generate a graph. By 
default, the whole stream is used. If provided, 
1 + (n_samples / samples_per_graph) will be generated;

- ` fc_measure`: functional connectivity measure to use;

- ` link_cutoff`: links with absolute FC measure below this value will be
removed;

- ` percentiles`: tuple of two numbers >0 and <100; links with FC measure 
between the two percentiles will be removed (statistics are calculated for
each edge). Note that this option ignores `link_cutoff`.

- ` band_freq_hi`: high band used to estimate FC when using 'aec';

- ` nfft`: TODO, affects 'wpli' and 'dwpli';

- ` n_overlap`: TODO, affects 'wpli' and 'dwpli';

- ` njobs`: number of processes to use (-1 to use all available cores);

**Return**  
 FC graph(s) in numpy format (note that node features are all ones).
