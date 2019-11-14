### generate_data


```python
spektral.datasets.delaunay.generate_data(classes=0, n_samples_in_class=1000, n_nodes=7, support_low=0.0, support_high=10.0, drift_amount=1.0, one_hot_labels=True, support=None, seed=None, return_type='numpy')
```



Generates a dataset of Delaunay triangulations as described by
[Zambon et al. (2017)](https://arxiv.org/abs/1706.06941).

Node attributes are the 2D coordinates of the points.
Two nodes are connected if they share an edge in the Delaunay triangulation.
Labels represent the class of the graph (0 to 20, each class index i
represent the "difficulty" of the classification problem 0 v. i. In other
words, the higher the class index, the more similar the class is to class 0).


**Arguments**  

- ` classes`: indices of the classes to load (integer, or list of integers
between 0 and 20);

- ` n_samples_in_class`: number of generated samples per class;

- ` n_nodes`: number of nodes in a graph;

- ` support_low`: lower bound of the uniform distribution from which the 
support is generated;

- ` support_high`: upper bound of the uniform distribution from which the 
support is generated;

- ` drift_amount`: coefficient to control the amount of change between 
classes;

- ` one_hot_labels`: one-hot encode dataset labels;

- ` support`: custom support to use instead of generating it randomly; 

- ` seed`: random numpy seed;

- ` return_type`: `'numpy'` or `'networkx'`, data format to return;

**Return**  

- if `return_type='numpy'`, the adjacency matrix, node features, and
an array containing labels;
- if `return_type='networkx'`, a list of graphs in Networkx format, and an
array containing labels;
