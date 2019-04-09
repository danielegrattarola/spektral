This module provides some functions to work with molecules, and requires
the [RDKit](http://www.rdkit.org/docs/index.html) library to be
installed on the system.

### numpy_to_rdkit


```python
spektral.chem.numpy_to_rdkit(adj, nf, ef, sanitize=False)
```



Converts a molecule from numpy to RDKit format.

**Arguments**  

- ` adj`: binary numpy array of shape (N, N) 

- ` nf`: numpy array of shape (N, F)

- ` ef`: numpy array of shape (N, N, S)

- ` sanitize`: whether to sanitize the molecule after conversion

**Return**  
 An RDKit molecule

----

### numpy_to_smiles


```python
spektral.chem.numpy_to_smiles(adj, nf, ef)
```



Converts a molecule from numpy to SMILES format.

**Arguments**  

- ` adj`: binary numpy array of shape (N, N) 

- ` nf`: numpy array of shape (N, F)

- ` ef`: numpy array of shape (N, N, S) 

**Return**  
 The SMILES string of the molecule

----

### rdkit_to_smiles


```python
spektral.chem.rdkit_to_smiles(mol)
```



Returns the SMILES string representing an RDKit molecule.

**Arguments**  

- ` mol`: an RDKit molecule

**Return**  
 The SMILES string of the molecule 

----

### sdf_to_nx


```python
spektral.chem.sdf_to_nx(sdf, keep_hydrogen=False)
```



Converts molecules in SDF format to networkx Graphs.

**Arguments**  

- ` sdf`: a list of molecules (or individual molecule) in SDF format.

- ` keep_hydrogen`: whether to include hydrogen in the representation.

**Return**  
 List of nx.Graphs.

----

### nx_to_sdf


```python
spektral.chem.nx_to_sdf(graphs)
```



Converts a list of nx.Graphs to the internal SDF format.

**Arguments**  

- ` graphs`: list of nx.Graphs.

**Return**  
 List of molecules in the internal SDF format.

----

### validate_rdkit


```python
spektral.chem.validate_rdkit(mol)
```



Validates RDKit molecules (single or in a list). 

**Arguments**  

- ` mol`: an RDKit molecule or list/np.array thereof

**Return**  
 Boolean array, True if the molecules are chemically valid, False 
otherwise

----

### get_atomic_symbol


```python
spektral.chem.get_atomic_symbol(number)
```



Given an atomic number (e.g., 6), returns its atomic symbol (e.g., 'C')

**Arguments**  

- ` number`: int <= 118

**Return**  
 String, atomic symbol

----

### get_atomic_num


```python
spektral.chem.get_atomic_num(symbol)
```



Given an atomic symbol (e.g., 'C'), returns its atomic number (e.g., 6)

**Arguments**  

- ` symbol`: string, atomic symbol

**Return**  
 Int <= 118

----

### valid_score


```python
spektral.chem.valid_score(molecules, from_numpy=False)
```



For a given list of molecules (RDKit or numpy format), returns a boolean 
array representing the validity of each molecule.

**Arguments**  

- ` molecules`: list of molecules (RDKit or numpy format)

- ` from_numpy`: whether the molecules are in numpy format

**Return**  
 Boolean array with the validity for each molecule

----

### novel_score


```python
spektral.chem.novel_score(molecules, smiles, from_numpy=False)
```



For a given list of molecules (RDKit or numpy format), returns a boolean 
array representing valid and novel molecules with respect to the list
of smiles provided (a molecule is novel if its SMILES is not in the list).

**Arguments**  

- ` molecules`: list of molecules (RDKit or numpy format)

- ` smiles`: list or set of smiles strings against which to check for 
novelty

- ` from_numpy`: whether the molecules are in numpy format

**Return**  
 Boolean array with the novelty for each valid molecule

----

### unique_score


```python
spektral.chem.unique_score(molecules, from_numpy=False)
```



For a given list of molecules (RDKit or numpy format), returns the fraction
of unique and valid molecules w.r.t. to the number of valid molecules.

**Arguments**  

- ` molecules`: list of molecules (RDKit or numpy format)

- ` from_numpy`: whether the molecules are in numpy format

**Return**  
 Fraction of unique valid molecules w.r.t. to valid molecules

----

### enable_rdkit_log


```python
spektral.chem.enable_rdkit_log()
```



Enables RDkit logging.

**Return**  


----

### plot_rdkit


```python
spektral.chem.plot_rdkit(mol, filename=None)
```



Plots an RDKit molecule in Matplotlib

**Arguments**  

- ` mol`: an RDKit molecule 

- ` filename`: save the image with the given filename 

**Return**  
 

----

### plot_rdkit_svg_grid


```python
spektral.chem.plot_rdkit_svg_grid(mols, mols_per_row=5, filename=None)
```



Plots a grid of RDKit molecules in SVG.

**Arguments**  

- ` mols`: a list of RDKit molecules

- ` mols_per_row`: size of the grid

- ` filename`: save an image with the given filename

- ` kwargs`: additional arguments for `RDKit.Chem.Draw.MolsToGridImage`

**Return**  
 The SVG as a string
