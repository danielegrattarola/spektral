import ast
import sys

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..chem import get_atomic_num


def load_binary(filename):
    """
    Loads a pickled file.
    :param filename: a string or file-like object
    :return: the loaded object
    """
    try:
        return joblib.load(filename)
    except ValueError:
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def dump_binary(obj, filename):
    """
    Pickles and saves an object to file.
    :param obj: the object to save
    :param filename: a string or file-like object
    """
    joblib.dump(obj, filename)


def load_csv(filename, **kwargs):
    """
    Loads a csv file with pandas.
    :param filename: a string or file-like object
    :return: the loaded csv
    """
    return pd.read_csv(filename, **kwargs)


def dump_csv(df, filename, convert=False, **kwargs):
    """
    Dumps a pd.DataFrame to csv.
    :param df: the pd.DataFrame to save or equivalent object
    :param filename: a string or file-like object
    :param convert: whether to attempt to convert the given object to
    pd.DataFrame before saving the csv.
    """
    if convert:
        df = pd.DataFrame(df)
    assert hasattr(df, 'to_csv'), \
        'Trying to dump object of class {} to csv while pd.DataFrame is ' \
        'expected. To attempt automatic conversion, set ' \
        'convert=True.'.format(df.__class__)
    df.to_csv(filename, **kwargs)


def load_dot(filename, force_graph=True):
    """
    Loads a graph saved in .dot format.
    :param filename: a string or file-like object
    :param force_graph: whether to force a conversion to nx.Graph after loading.
    This may be useful in the case of .dot files being loaded as nx.MultiGraph.
    :return: the loaded graph
    """
    output = nx.nx_agraph.read_dot(filename)
    if force_graph:
        output = nx.Graph(output)

    for elem in output.nodes().values():
        for k, v in elem.items():
            try:
                elem[k] = ast.literal_eval(v)
            except ValueError:
                elem[k] = str(v)
            except SyntaxError:
                # Probably a numpy array
                elem[k] = np.array(' '.join(v.lstrip('[')
                                             .rstrip(']')
                                             .split())
                                      .split(' ')).astype(np.float)

    for elem in output.edges().values():
        for k, v in elem.items():
            try:
                elem[k] = ast.literal_eval(v)
            except ValueError:
                elem[k] = str(v)

    return output


def dump_dot(obj, filename):
    """
    Dumps a nx.Graph to .dot file
    :param obj: the nx.Graph (or equivalent) to save
    :param filename: a string or file-like object
    """
    nx.nx_agraph.write_dot(obj, filename)


def load_npy(filename):
    """
    Loads a file saved by np.save.
    :param filename: a string or file-like object
    :return: the loaded object
    """
    if sys.version_info[0] == 3:
        return np.load(filename, encoding='latin1')
    else:
        return np.load(filename)


def dump_npy(obj, filename, zipped=False):
    """
    Saves an object to file using the numpy format.
    :param obj: the object to save
    :param filename: a string or file-like object
    :param zipped: boolean, whether to save the object in the zipped format .npz
    rather than .npy
    """
    if zipped:
        np.savez(filename, obj)
    else:
        np.save(filename, obj)


def load_txt(filename, **kwargs):
    """
    Loads a txt file using np.genfromtxt.
    :param filename: a string or file-like object
    :return: the loaded object
    """
    return np.genfromtxt(filename, **kwargs)


def dump_txt(obj, filename, **kwargs):
    """
    Saves an object to text file using np.savetxt.
    :param obj: the object to save
    :param filename: a string or file-like object
    """
    np.savetxt(filename, obj, **kwargs)


# Reference for implementation:
# # http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
#
# While parsing the SDF file, molecules are stored in a dictionary like this:
#
# {'atoms': [{'atomic_num': 7,
#             'charge': 0,
#             'coords': array([-0.0299,  1.2183,  0.2994]),
#             'index': 0,
#             'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             'iso': 0},
#            ...,
#            {'atomic_num': 1,
#             'charge': 0,
#             'coords': array([ 0.6896, -2.3002, -0.1042]),
#             'index': 14,
#             'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             'iso': 0}],
#  'bonds': [{'end_atom': 13,
#             'info': array([0, 0, 0]),
#             'start_atom': 4,
#             'stereo': 0,
#             'type': 1},
#            ...,
#            {'end_atom': 8,
#             'info': array([0, 0, 0]),
#             'start_atom': 7,
#             'stereo': 0,
#             'type': 3}],
#  'comment': '',
#  'data': [''],
#  'details': '-OEChem-03231823253D',
#  'n_atoms': 15,
#  'n_bonds': 15,
#  'name': 'gdb_54964',
#  'properties': []}
HEADER_SIZE = 3


def _parse_header(sdf):
    try:
        return sdf[0].strip(), sdf[1].strip(), sdf[2].strip()
    except IndexError:
        print(sdf)


def _parse_counts_line(sdf):
    # 12 fields
    # First 11 are 3 characters long
    # Last one is 6 characters long
    # First two give the number of atoms and bonds

    values = sdf[HEADER_SIZE]
    n_atoms = int(values[:3])
    n_bonds = int(values[3:6])

    return n_atoms, n_bonds


def _parse_atoms_block(sdf, n_atoms):
    # The first three fields, 10 characters long each, describe the atom's
    # position in the X, Y, and Z dimensions.
    # After that there is a space, and three characters for an atomic symbol.
    # After the symbol, there are two characters for the mass difference from
    # the monoisotope.
    # Next you have three characters for the charge.
    # There are ten more fields with three characters each, but these are all
    # rarely used.

    start = HEADER_SIZE + 1  # Add 1 for counts line
    stop = start + n_atoms
    values = sdf[start:stop]

    atoms = []
    for i, v in enumerate(values):
        coords = np.array([float(v[pos:pos+10]) for pos in range(0, 30, 10)])
        atomic_num = get_atomic_num(v[31:34].strip())
        iso = int(v[34:36])
        charge = int(v[36:39])
        info = np.array([int(v[pos:pos+3]) for pos in range(39, len(v), 3)])
        atoms.append({'index': i,
                      'coords': coords,
                      'atomic_num': atomic_num,
                      'iso': iso,
                      'charge': charge,
                      'info': info})
    return atoms


def _parse_bonds_block(sdf, n_atoms, n_bonds):
    # The first two fields are the indexes of the atoms included in this bond
    # (starting from 1). The third field defines the type of bond, and the
    # fourth the stereoscopy of the bond.
    # There are a further three fields, with 3 characters each, but these are
    # rarely used and can be left blank.

    start = HEADER_SIZE + n_atoms + 1  # Add 1 for counts line
    stop = start + n_bonds
    values = sdf[start:stop]

    bonds = []
    for v in values:
        start_atom = int(v[:3]) - 1
        end_atom = int(v[3:6]) - 1
        type_ = int(v[6:9])
        stereo = int(v[9:12])
        info = np.array([int(v[pos:pos + 3]) for pos in range(12, len(v), 3)])
        bonds.append({'start_atom': start_atom,
                      'end_atom': end_atom,
                      'type': type_,
                      'stereo': stereo,
                      'info': info})
    return bonds


def _parse_properties(sdf, n_atoms, n_bonds):
    # TODO This should be implemented properly, for now it just returns a list of properties.
    # See https://docs.chemaxon.com/display/docs/MDL+MOLfiles%2C+RGfiles%2C+SDfiles%2C+Rxnfiles%2C+RDfiles+formats
    # for documentation.

    start = HEADER_SIZE + n_atoms + n_bonds + 1  # Add 1 for counts line
    stop = sdf.index('M  END')

    return sdf[start:stop]


def _parse_data_fields(sdf):
    # TODO This should be implemented properly, for now it just returns a list of data fields.

    start = sdf.index('M  END') + 1

    return sdf[start:] if start < len(sdf) else []


def parse_sdf(sdf):
    sdf_out = {}
    sdf = sdf.split('\n')
    sdf_out['name'], sdf_out['details'], sdf_out['comment'] = _parse_header(sdf)
    sdf_out['n_atoms'], sdf_out['n_bonds'] = _parse_counts_line(sdf)
    sdf_out['atoms'] = _parse_atoms_block(sdf, sdf_out['n_atoms'])
    sdf_out['bonds'] = _parse_bonds_block(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['properties'] = _parse_properties(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['data'] = _parse_data_fields(sdf)
    return sdf_out


def parse_sdf_file(sdf_file, amount=None):
    data = sdf_file.read().split('$$$$\n')
    if data[-1] == '':
        data = data[:-1]
    if amount is not None:
        data = data[:amount]
    output = [parse_sdf(sdf) for sdf in tqdm(data)]  # Parallel execution doesn't help
    return output


def load_sdf(filename, amount=None):
    """
    Load an .sdf file and return a list of molecules in the internal SDF format.
    :param filename: target SDF file
    :param amount: only load the first `amount` molecules from the file
    :return: a list of molecules in the internal SDF format (see documentation).
    """
    print('Reading SDF')
    with open(filename) as f:
        return parse_sdf_file(f, amount=amount)
