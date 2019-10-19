import networkx as nx
import numpy as np
try:
    from rdkit import Chem as rdc
    from rdkit.Chem import Draw
    from rdkit import rdBase as rdb

    rdb.DisableLog('rdApp.error')  # RDKit logging is disabled by default
    Draw.DrawingOptions.dblBondOffset = .1
    BOND_MAP = {0: rdc.rdchem.BondType.ZERO,
                1: rdc.rdchem.BondType.SINGLE,
                2: rdc.rdchem.BondType.DOUBLE,
                3: rdc.rdchem.BondType.TRIPLE,
                4: rdc.rdchem.BondType.AROMATIC}
except ImportError:
    rdc = None
    rdb = None

NUM_TO_SYMBOL = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
                 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
                 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K',
                 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
                 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
                 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
                 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
                 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs',
                 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
                 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
                 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
                 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
                 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
                 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
                 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk',
                 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr',
                 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs',
                 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh',
                 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}
SYMBOL_TO_NUM = {v: k for k, v in NUM_TO_SYMBOL.items()}


def numpy_to_rdkit(adj, nf, ef, sanitize=False):
    """
    Converts a molecule from numpy to RDKit format.
    :param adj: binary numpy array of shape (N, N) 
    :param nf: numpy array of shape (N, F)
    :param ef: numpy array of shape (N, N, S)
    :param sanitize: whether to sanitize the molecule after conversion
    :return: an RDKit molecule
    """
    if rdc is None:
        raise ImportError('`numpy_to_rdkit` requires RDKit.')
    mol = rdc.RWMol()
    for nf_ in nf:
        atomic_num = int(nf_)
        if atomic_num > 0:
            mol.AddAtom(rdc.Atom(atomic_num))

    for i, j in zip(*np.triu_indices(adj.shape[-1])):
        if i != j and adj[i, j] == adj[j, i] == 1 and not mol.GetBondBetweenAtoms(int(i), int(j)):
            bond_type_1 = BOND_MAP[int(ef[i, j, 0])]
            bond_type_2 = BOND_MAP[int(ef[j, i, 0])]
            if bond_type_1 == bond_type_2:
                mol.AddBond(int(i), int(j), bond_type_1)

    mol = mol.GetMol()
    if sanitize:
        rdc.SanitizeMol(mol)
    return mol


def numpy_to_smiles(adj, nf, ef):
    """
    Converts a molecule from numpy to SMILES format.
    :param adj: binary numpy array of shape (N, N) 
    :param nf: numpy array of shape (N, F)
    :param ef: numpy array of shape (N, N, S) 
    :return: the SMILES string of the molecule
    """
    if rdc is None:
        raise ImportError('`numpy_to_smiles` requires RDkit.')
    mol = numpy_to_rdkit(adj, nf, ef)
    return rdkit_to_smiles(mol)


def rdkit_to_smiles(mol):
    """
    Returns the SMILES string representing an RDKit molecule.
    :param mol: an RDKit molecule
    :return: the SMILES string of the molecule 
    """
    if rdc is None:
        raise ImportError('`rdkit_to_smiles` requires RDkit.')
    return rdc.MolToSmiles(mol)


def sdf_to_nx(sdf, keep_hydrogen=False):
    """
    Converts molecules in SDF format to networkx Graphs.
    :param sdf: a list of molecules (or individual molecule) in SDF format.
    :param keep_hydrogen: whether to include hydrogen in the representation.
    :return: list of nx.Graphs.
    """
    if not isinstance(sdf, list):
        sdf = [sdf]

    output = []
    for sdf_ in sdf:
        g = nx.Graph()

        for atom in sdf_['atoms']:
            if atom['atomic_num'] > 1 or keep_hydrogen:
                g.add_node(atom['index'], **atom)
        for bond in sdf_['bonds']:
            start_atom_num = sdf_['atoms'][bond['start_atom']]['atomic_num']
            end_atom_num = sdf_['atoms'][bond['end_atom']]['atomic_num']
            if (start_atom_num > 1 and end_atom_num > 1) or keep_hydrogen:
                g.add_edge(bond['start_atom'], bond['end_atom'], **bond)
        output.append(g)

    if len(output) == 1:
        return output[0]
    else:
        return output


def nx_to_sdf(graphs):
    """
    Converts a list of nx.Graphs to the internal SDF format.
    :param graphs: list of nx.Graphs.
    :return: list of molecules in the internal SDF format.
    """
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]
    output = []
    for g in graphs:
        sdf = {'atoms': [v for k, v in g.nodes.items()],
               'bonds': [v for k, v in g.edges.items()],
               'comment': '',
               'data': [''],
               'details': '',
               'n_atoms': -1,
               'n_bonds': -1,
               'name': '',
               'properties': []}
        output.append(sdf)
    return output


def validate_rdkit_mol(mol):
    """
    Sanitizes an RDKit molecules and returns True if the molecule is chemically
    valid.
    :param mol: an RDKit molecule 
    :return: True if the molecule is chemically valid, False otherwise
    """
    if rdc is None:
        raise ImportError('`validate_rdkit_mol` requires RDkit.')
    if len(rdc.GetMolFrags(mol)) > 1:
        return False
    try:
        rdc.SanitizeMol(mol)
        return True
    except ValueError:
        return False


def validate_rdkit(mol):
    """
    Validates RDKit molecules (single or in a list). 
    :param mol: an RDKit molecule or list/np.array thereof
    :return: boolean array, True if the molecules are chemically valid, False 
    otherwise
    """
    if rdc is None:
        raise ImportError('`validate_rdkit` requires RDkit.')
    if isinstance(mol, list) or isinstance(mol, np.ndarray):
        return np.array([validate_rdkit_mol(m) for m in mol])
    else:
        return validate_rdkit_mol(mol)


def get_atomic_symbol(number):
    """
    Given an atomic number (e.g., 6), returns its atomic symbol (e.g., 'C')
    :param number: int <= 118
    :return: string, atomic symbol
    """
    return NUM_TO_SYMBOL[number]


def get_atomic_num(symbol):
    """
    Given an atomic symbol (e.g., 'C'), returns its atomic number (e.g., 6)
    :param symbol: string, atomic symbol
    :return: int <= 118
    """
    return SYMBOL_TO_NUM[symbol.lower().capitalize()]


def valid_score(molecules, from_numpy=False):
    """
    For a given list of molecules (RDKit or numpy format), returns a boolean 
    array representing the validity of each molecule.
    :param molecules: list of molecules (RDKit or numpy format)
    :param from_numpy: whether the molecules are in numpy format
    :return: boolean array with the validity for each molecule
    """
    if rdc is None:
        raise ImportError('`valid_score` requires RDkit.')
    valid = []
    if from_numpy:
        molecules = [numpy_to_rdkit(adj_p, nf_p, ef_p)
                     for adj_p, nf_p, ef_p in molecules]
    for mol_rdk in molecules:
        valid.append(validate_rdkit_mol(mol_rdk))

    return np.array(valid)


def novel_score(molecules, smiles, from_numpy=False):
    """
    For a given list of molecules (RDKit or numpy format), returns a boolean 
    array representing valid and novel molecules with respect to the list
    of smiles provided (a molecule is novel if its SMILES is not in the list).
    :param molecules: list of molecules (RDKit or numpy format)
    :param smiles: list or set of smiles strings against which to check for 
    novelty
    :param from_numpy: whether the molecules are in numpy format
    :return: boolean array with the novelty for each valid molecule
    """
    if rdc is None:
        raise ImportError('`novel_score` requires RDkit.')
    if from_numpy:
        molecules = [numpy_to_rdkit(adj_p, nf_p, ef_p)
                     for adj_p, nf_p, ef_p in molecules]
    smiles = set(smiles)
    novel = []
    for mol in molecules:
        is_valid = validate_rdkit_mol(mol)
        is_novel = rdkit_to_smiles(mol) not in smiles
        novel.append(is_valid and is_novel)

    return np.array(novel)


def unique_score(molecules, from_numpy=False):
    """
    For a given list of molecules (RDKit or numpy format), returns the fraction
    of unique and valid molecules w.r.t. to the number of valid molecules.
    :param molecules: list of molecules (RDKit or numpy format)
    :param from_numpy: whether the molecules are in numpy format
    :return: fraction of unique valid molecules w.r.t. to valid molecules
    """
    if rdc is None:
        raise ImportError('`unique_score` requires RDkit.')
    if from_numpy:
        molecules = [numpy_to_rdkit(adj_p, nf_p, ef_p)
                     for adj_p, nf_p, ef_p in molecules]
    smiles = set()
    n_valid = 0
    for mol in molecules:
        if validate_rdkit_mol(mol):
            n_valid += 1
            smiles.add(rdkit_to_smiles(mol))

    return 0 if n_valid == 0 else (len(smiles) / n_valid)


def enable_rdkit_log():
    """
    Enables RDkit logging.
    :return:
    """
    if rdb is None:
        raise ImportError('`enable_rdkit_log` requires RDkit.')
    rdb.EnableLog('rdApp.error')


def plot_rdkit(mol, filename=None):
    """
    Plots an RDKit molecule in Matplotlib
    :param mol: an RDKit molecule 
    :param filename: save the image with the given filename 
    :return: the image as np.array
    """
    if rdc is None:
        raise ImportError('`draw_rdkit_mol` requires RDkit.')
    if filename is not None:
        Draw.MolToFile(mol, filename)
    img = Draw.MolToImage(mol)
    return img


def plot_rdkit_svg_grid(mols, mols_per_row=5, filename=None, **kwargs):
    """
    Plots a grid of RDKit molecules in SVG.
    :param mols: a list of RDKit molecules
    :param mols_per_row: size of the grid
    :param filename: save an image with the given filename
    :param kwargs: additional arguments for `RDKit.Chem.Draw.MolsToGridImage`
    :return: the SVG as a string
    """
    if rdc is None:
        raise ImportError('`draw_rdkit_mol` requires RDkit.')
    svg = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, useSVG=True, **kwargs)
    if filename is not None:
        if not filename.endswith('.svg'):
            filename += '.svg'
        with open(filename, 'w') as f:
            f.write(svg)
    return svg

