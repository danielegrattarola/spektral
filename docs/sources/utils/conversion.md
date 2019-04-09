### nx_to_adj


```python
spektral.utils.nx_to_adj(graphs)
```



Converts a list of nx.Graphs to a rank 3 np.array of adjacency matrices
of shape `(num_graphs, num_nodes, num_nodes)`.

**Arguments**  

- ` graphs`: a nx.Graph, or list of nx.Graphs.

**Return**  
 A rank 3 np.array of adjacency matrices.

----

### nx_to_node_features


```python
spektral.utils.nx_to_node_features(graphs, keys, post_processing=None)
```



Converts a list of nx.Graphs to a rank 3 np.array of node features matrices
of shape `(num_graphs, num_nodes, num_features)`. Optionally applies a
post-processing function to each individual attribute in the nx Graphs.

**Arguments**  

- ` graphs`: a nx.Graph, or a list of nx.Graphs;

- ` keys`: a list of keys with which to index node attributes in the nx
Graphs.

- ` post_processing`: a list of functions with which to post process each
attribute associated to a key. `None` can be passed as post-processing 
function to leave the attribute unchanged.

**Return**  
 A rank 3 np.array of feature matrices

----

### nx_to_edge_features


```python
spektral.utils.nx_to_edge_features(graphs, keys, post_processing=None)
```



Converts a list of nx.Graphs to a rank 4 np.array of edge features matrices
of shape `(num_graphs, num_nodes, num_nodes, num_features)`.
Optionally applies a post-processing function to each attribute in the nx
graphs.

**Arguments**  

- ` graphs`: a nx.Graph, or a list of nx.Graphs;

- ` keys`: a list of keys with which to index edge attributes.

- ` post_processing`: a list of functions with which to post process each
attribute associated to a key. `None` can be passed as post-processing 
function to leave the attribute unchanged.

**Return**  
 A rank 3 np.array of feature matrices

----

### nx_to_numpy


```python
spektral.utils.nx_to_numpy(graphs, auto_pad=True, self_loops=True, nf_keys=None, ef_keys=None, nf_postprocessing=None, ef_postprocessing=None)
```



Converts a list of nx.Graphs to numpy format (adjacency, node attributes,
and edge attributes matrices).

**Arguments**  

- ` graphs`: a nx.Graph, or list of nx.Graphs;

- ` auto_pad`: whether to zero-pad all matrices to have graphs with the
same dimension (set this to true if you don't want to deal with manual
batching for different-size graphs.

- ` self_loops`: whether to add self-loops to the graphs.

- ` nf_keys`: a list of keys with which to index node attributes. If None,
returns None as node attributes matrix.

- ` ef_keys`: a list of keys with which to index edge attributes. If None,
returns None as edge attributes matrix.

- ` nf_postprocessing`: a list of functions with which to post process each
node attribute associated to a key. `None` can be passed as post-processing
function to leave the attribute unchanged.

- ` ef_postprocessing`: a list of functions with which to post process each
edge attribute associated to a key. `None` can be passed as post-processing
function to leave the attribute unchanged.

**Return**  

- adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`
- node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`
- edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`

----

### numpy_to_nx


```python
spektral.utils.numpy_to_nx(adj, node_features=None, edge_features=None, nf_name=None, ef_name=None)
```



Converts graphs in numpy format to a list of nx.Graphs.

**Arguments**  

- ` adj`: adjacency matrices of shape `(num_samples, num_nodes, num_nodes)`.
If there is only one sample, the first dimension can be dropped.

- ` node_features`: optional node attributes matrices of shape `(num_samples, num_nodes, node_features_dim)`.
If there is only one sample, the first dimension can be dropped.

- ` edge_features`: optional edge attributes matrices of shape `(num_samples, num_nodes, num_nodes, edge_features_dim)`
If there is only one sample, the first dimension can be dropped.

- ` nf_name`: optional name to assign to node attributes in the nx.Graphs

- ` ef_name`: optional name to assign to edge attributes in the nx.Graphs

**Return**  
 A list of nx.Graphs (or a single nx.Graph is there is only one sample)
