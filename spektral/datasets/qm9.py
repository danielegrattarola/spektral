from __future__ import absolute_import

import os

from keras.utils import get_file

from spektral.chem import sdf_to_nx
from spektral.utils.io import load_csv, load_sdf
from spektral.utils import nx_to_numpy

DATA_PATH = os.path.expanduser('~/.spektral/datasets/qm9/')
DATASET_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
RETURN_TYPES = {'numpy', 'networkx', 'sdf'}
NODE_FEATURES = ['atomic_num', 'charge', 'coords', 'iso']
EDGE_FEATURES = ['type', 'stereo']
MAX_K = 9


def load_data(return_type='numpy', nf_keys=None, ef_keys=None, auto_pad=True,
              self_loops=False, amount=None):
    """
    Loads the QM9 molecules dataset.
    :param return_type: 'networkx', 'numpy', or 'sdf', data format to return;
    :param nf_keys: list or str, node features to return (see `qm9.NODE_FEATURES`
    for available features);
    :param ef_keys: list or str, edge features to return (see `qm9.EDGE_FEATURES`
    for available features);
    :param auto_pad: if `return_type='numpy'`, zero pad graph matrices to have 
    the same number of nodes;
    :param self_loops: if `return_type='numpy'`, add self loops to adjacency 
    matrices;
    :param amount: the amount of molecules to return (in order).
    :return: if `return_type='numpy'`, the adjacency matrix, node features,
    edge features, and a Pandas dataframe containing labels;
    if `return_type='networkx'`, a list of graphs in Networkx format,
    and a dataframe containing labels;   
    if `return_type='sdf'`, a list of molecules in the internal SDF format and
    a dataframe containing labels.
    """
    if return_type not in RETURN_TYPES:
        raise ValueError('Possible return_type: {}'.format(RETURN_TYPES))

    if not os.path.exists(DATA_PATH):
        _ = dataset_downloader()  # Try to download dataset

    print('Loading QM9 dataset.')
    sdf_file = os.path.join(DATA_PATH, 'qm9.sdf')
    data = load_sdf(sdf_file, amount=amount)  # Internal SDF format

    # Load labels
    labels_file = os.path.join(DATA_PATH, 'qm9.sdf.csv')
    labels = load_csv(labels_file)
    if amount is not None:
        labels = labels[:amount]
    if return_type is 'sdf':
        return data, labels
    else:
        # Convert to Networkx
        data = [sdf_to_nx(_) for _ in data]

    if return_type is 'numpy':
        if nf_keys is not None:
            if isinstance(nf_keys, str):
                nf_keys = [nf_keys]
        else:
            nf_keys = NODE_FEATURES
        if ef_keys is not None:
            if isinstance(ef_keys, str):
                ef_keys = [ef_keys]
        else:
            ef_keys = EDGE_FEATURES

        adj, nf, ef = nx_to_numpy(data,
                                  auto_pad=auto_pad, self_loops=self_loops,
                                  nf_keys=nf_keys, ef_keys=ef_keys)
        return adj, nf, ef, labels
    elif return_type is 'networkx':
        return data, labels
    else:
        # Should not get here
        raise RuntimeError()


def dataset_downloader():
    filename = get_file('qm9.tar.gz', DATASET_URL, extract=True,
                        cache_dir=DATA_PATH, cache_subdir=DATA_PATH)
    os.rename(DATA_PATH + 'gdb9.sdf', DATA_PATH + 'qm9.sdf')
    os.rename(DATA_PATH + 'gdb9.sdf.csv', DATA_PATH + 'qm9.sdf.csv')
    os.remove(DATA_PATH + 'qm9.tar.gz')
    return filename
