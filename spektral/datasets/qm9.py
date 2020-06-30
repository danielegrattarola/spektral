import os

from tensorflow.keras.utils import get_file

from spektral.chem import sdf_to_nx
from spektral.utils import nx_to_numpy
from spektral.utils.io import load_csv, load_sdf

DATA_PATH = os.path.expanduser('~/.spektral/datasets/qm9/')
DATASET_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
RETURN_TYPES = {'numpy', 'networkx', 'sdf'}
NODE_FEATURES = ['atomic_num', 'charge', 'coords', 'iso']
EDGE_FEATURES = ['type', 'stereo']
MAX_K = 9


def load_data(nf_keys=None, ef_keys=None, auto_pad=True, self_loops=False,
              amount=None, return_type='numpy'):
    """
    Loads the QM9 chemical data set of small molecules.

    Nodes represent heavy atoms (hydrogens are discarded), edges represent
    chemical bonds.

    The node features represent the chemical properties of each atom, and are
    loaded according to the `nf_keys` argument.
    See `spektral.datasets.qm9.NODE_FEATURES` for possible node features, and
    see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
    for the meaning of each property. Usually, it is sufficient to load the
    atomic number.

    The edge features represent the type and stereoscopy of each chemical bond
    between two atoms.
    See `spektral.datasets.qm9.EDGE_FEATURES` for possible edge features, and
    see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
    for the meaning of each property. Usually, it is sufficient to load the
    type of bond.

    :param nf_keys: list or str, node features to return (see `qm9.NODE_FEATURES`
    for available features);
    :param ef_keys: list or str, edge features to return (see `qm9.EDGE_FEATURES`
    for available features);
    :param auto_pad: if `return_type='numpy'`, zero pad graph matrices to have 
    the same number of nodes;
    :param self_loops: if `return_type='numpy'`, add self loops to adjacency 
    matrices;
    :param amount: the amount of molecules to return (in ascending order by
    number of atoms).
    :param return_type: `'numpy'`, `'networkx'`, or `'sdf'`, data format to return;
    :return:
    - if `return_type='numpy'`, the adjacency matrix, node features,
    edge features, and a Pandas dataframe containing labels;
    - if `return_type='networkx'`, a list of graphs in Networkx format,
    and a dataframe containing labels;   
    - if `return_type='sdf'`, a list of molecules in the internal SDF format and
    a dataframe containing labels.
    """
    if return_type not in RETURN_TYPES:
        raise ValueError('Possible return_type: {}'.format(RETURN_TYPES))

    if not os.path.exists(DATA_PATH):
        _download_data()  # Try to download dataset

    print('Loading QM9 dataset.')
    sdf_file = os.path.join(DATA_PATH, 'qm9.sdf')
    data = load_sdf(sdf_file, amount=amount)  # Internal SDF format

    # Load labels
    labels_file = os.path.join(DATA_PATH, 'qm9.sdf.csv')
    labels = load_csv(labels_file)
    if amount is not None:
        labels = labels[:amount]
    if return_type == 'sdf':
        return data, labels
    else:
        # Convert to Networkx
        data = [sdf_to_nx(_) for _ in data]

    if return_type == 'numpy':
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
    elif return_type == 'networkx':
        return data, labels
    else:
        # Should not get here
        raise RuntimeError()


def _download_data():
    _ = get_file(
        'qm9.tar.gz', DATASET_URL,
        extract=True, cache_dir=DATA_PATH, cache_subdir=DATA_PATH
    )
    os.rename(DATA_PATH + 'gdb9.sdf', DATA_PATH + 'qm9.sdf')
    os.rename(DATA_PATH + 'gdb9.sdf.csv', DATA_PATH + 'qm9.sdf.csv')
    os.remove(DATA_PATH + 'qm9.tar.gz')
