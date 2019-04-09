### load_data


```python
spektral.datasets.qm9.load_data(return_type='numpy', nf_keys=None, ef_keys=None, auto_pad=True, self_loops=False, amount=None)
```



Loads the QM9 molecules dataset.

**Arguments**  

- ` return_type`: 'networkx', 'numpy', or 'sdf', data format to return;

- ` nf_keys`: list or str, node features to return (see `qm9.NODE_FEATURES`
for available features);

- ` ef_keys`: list or str, edge features to return (see `qm9.EDGE_FEATURES`
for available features);

- ` auto_pad`: if `return_type='numpy'`, zero pad graph matrices to have 
the same number of nodes;

- ` self_loops`: if `return_type='numpy'`, add self loops to adjacency 
matrices;

- ` amount`: the amount of molecules to return (in order).

**Return**  
 If `return_type='numpy'`, the adjacency matrix, node features,
edge features, and a Pandas dataframe containing labels;
if `return_type='networkx'`, a list of graphs in Networkx format,
and a dataframe containing labels;   
if `return_type='sdf'`, a list of molecules in the internal SDF format and
a dataframe containing labels.
