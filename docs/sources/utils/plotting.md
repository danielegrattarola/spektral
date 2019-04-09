### plot_numpy


```python
spektral.utils.plot_numpy(adj, node_features=None, edge_features=None, nf_name=None, ef_name=None, layout='spring_layout', labels=True, node_color='r', node_size=300)
```



Converts a graph in matrix format (i.e. with adjacency matrix, node features
matrix, and edge features matrix) to the Networkx format, then plots it with
plot_nx().

**Arguments**  

- ` adj`: np.array, adjacency matrix of the graph 

- ` node_features`: np.array, node features matrix of the graph

- ` edge_features`: np.array, edge features matrix of the graph

- ` nf_name`: name to assign to the node features

- ` ef_name`: name to assign to the edge features

- ` layout`: type of layout for networkx

- ` labels`: plot labels

- ` node_color`: color for the plotted nodes

- ` node_size`: size of the plotted nodes

**Return**  
 None

----

### plot_nx


```python
spektral.utils.plot_nx(nx_graph, nf_name=None, ef_name=None, layout='spring_layout', labels=True, node_color='r', node_size=300)
```



Plot the given Networkx graph.

**Arguments**  

- ` nx_graph`: a Networkx graph

- ` nf_name`: name of the relevant node feature to plot

- ` ef_name`: name of the relevant edgee feature to plot

- ` layout`: type of layout for networkx

- ` labels`: plot labels

- ` node_color`: color for the plotted nodes

- ` node_size`: size of the plotted nodes

**Return**  
 None
