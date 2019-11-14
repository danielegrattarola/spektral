### load_data


```python
spektral.datasets.qm9.load_data(nf_keys=None, ef_keys=None, auto_pad=True, self_loops=False, amount=None, return_type='numpy')
```



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


**Arguments**  

- ` nf_keys`: list or str, node features to return (see `qm9.NODE_FEATURES`
for available features);

- ` ef_keys`: list or str, edge features to return (see `qm9.EDGE_FEATURES`
for available features);

- ` auto_pad`: if `return_type='numpy'`, zero pad graph matrices to have 
the same number of nodes;

- ` self_loops`: if `return_type='numpy'`, add self loops to adjacency 
matrices;

- ` amount`: the amount of molecules to return (in ascending order by
number of atoms).

- ` return_type`: `'numpy'`, `'networkx'`, or `'sdf'`, data format to return;

**Return**  

- if `return_type='numpy'`, the adjacency matrix, node features,
edge features, and a Pandas dataframe containing labels;
- if `return_type='networkx'`, a list of graphs in Networkx format,
and a dataframe containing labels;   
- if `return_type='sdf'`, a list of molecules in the internal SDF format and
a dataframe containing labels.
