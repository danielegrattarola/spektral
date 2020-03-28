## Representing graphs

Spektral uses a matrix-based representation for manipulating graphs and feeding them to neural networks. This approach is one of the most commonly used in the literature on graph neural networks, and it's perfect to perform parallel computations on GPU.


A graph is generally represented by three matrices:

- \(A \in \{0, 1\}^{N \times N}\), a square adjacency matrix where \(A_{ij} = 1\) if there is a connection between nodes \(i\) and \(j\), and \(A_{ij} = 0\) otherwise;
- \(X \in \mathbb{R}^{N \times F}\), a matrix encoding node attributes, where each row represents the \(F\)-dimensional attribute vector of a node;
- \(E \in \mathbb{R}^{N \times N \times S}\), a matrix encoding edge attributes, where each entry represents the \(S\)-dimensional attribute vector of an edge;

Some frameworks (like the [graph networks](https://arxiv.org/abs/1806.01261) proposed by Battaglia et al.) also include a feature vector describing the global state of the graph, but this is not supported by Spektral for now. 

In code, and in this documentation, we use the following convention to refer to the formulation above:

- `A` is the adjacency matrix, `N` is the number of nodes;
- `X` is the node attributes matrix, `F` is the size of the node attributes;
- `E` is the edge attributes matrix, `S` is the size of the edge attributes;

See the table below for how these matrices are represented in Numpy. 


## Modes

In Spektral, some functionalities are implemented to work on a single graph, while others consider batches of graphs. 

To understand the difference between the two settings, consider the difference between classifying the nodes of a citation network, and classifying the chemical properties of molecules.  
For the citation network, we are interested in the individual nodes and the connections between them. Node and edge attributes are specific to each individual network, and we are usually not interested in training models that work on different networks. The nodes themselves are our data.  
On the other hand, when working with molecules in a dataset, we are in a much more familiar setting. Each molecule is a sample of our dataset, and the atoms and bonds that make up the molecules are the constituent part of each data point (like pixels in images). In this case, we are interested in finding patterns that describe the properties of the molecules in general. 

The two settings require us to do things that are conceptually similar, but that need some minor adjustments in how the data is processed by our graph neural networks. This is why Spektral makes these differences explicit.

In practice, we actually distinguish between three main **modes** of operation: 

- **single**, where we have a single graph, with fixed topology and attributes;
- **batch**, where we have a set of different graphs, each with its own topology and attributes;
- **mixed**, where we have a graph with fixed topology, but a set of different attributes (usually called _graph signals_); this can be seen as a particular case of the batch mode, but it is handled separately in Spektral to improve memory efficiency.

We also have the **disjoint** mode, which is a simple trick to represent a batch of graphs in single mode. This requires an additional data structure to keep track of the graphs, and is explained in detail at the end of this section.

The difference between the three main modes can be easily seen in how `A`, `X`, and `E` have different shapes in each case:

|Mode  | `A.shape`     | `X.shape`     | `E.shape`        |
|:----:|:-------------:|:-------------:|:----------------:|
|Single|`(N, N)`       |`(N, F)`       |`(N, N, S)`       |
|Batch |`(batch, N, N)`|`(batch, N, F)`|`(batch, N, N, S)`|
|Mixed |`(N, N)`       |`(batch, N, F)`|`(batch, N, N, S)`|


### Single mode
In **single mode** the data describes a single graph. Three very popular datasets in this setting are the citation networks, Cora, Citeseer, and Pubmed. To load a citation network, you can use the built-in loader:

```py
In [1]: from spektral.datasets import citation
Using TensorFlow backend.

In [2]: A, X, _, _, _, _ = citation.load_data('cora')
Loading cora dataset

In [3]: A.shape
Out[3]: (2708, 2708)

In [4]: X.shape
Out[4]: (2708, 1433)
```

When training GNNs in single mode, we cannot batch and shuffle the data along the first axis, and the whole graph must be fed to the model at each step (see [the node classification example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_classification_gcn.py)).

### Batch mode
In **batch mode**, the matrices will have a `batch` dimension first. For instance, we can load the QM9 chemical database of small molecules as follows:

```py
In [1]: from spektral.datasets import qm9
Using TensorFlow backend.

In [2]: A, X, E, _ = qm9.load_data()
Loading QM9 dataset.
Reading SDF
100%|█████████████████████| 133885/133885 [00:29<00:00, 4579.22it/s]

In [3]: A.shape
Out[3]: (133885, 9, 9)

In [4]: X.shape
Out[4]: (133885, 9, 6)

In [5]: E.shape
Out[5]: (133885, 9, 9, 1)
```

Note that the graphs in QM9 have variable order (i.e., a different number of nodes for each graph), and that by default `load_data()` pads them with zeros in order to store the data in Numpy arrays.  
See the [disjoint mode](https://danielegrattarola.github.io/spektral/data/#disjoint-mode) section for an alternative to zero-padding. 

### Mixed mode
In **mixed mode** we consider a single adjacency matrix, and different node and edge attributes matrices. An example of a mixed mode dataset is the MNIST random grid proposed by [Defferrard et al.](https://arxiv.org/abs/1606.09375):

```py
In [1]: from spektral.datasets import mnist
Using TensorFlow backend.

In [2]: X, _, _, _, _, _, A = mnist.load_data()

In [3]: A.shape
Out[3]: (784, 784)

In [4]: X.shape
Out[4]: (50000, 784, 1)
```

### Disjoint mode

![](https://danielegrattarola.github.io/spektral/img/disjoint.svg)

When dealing with graphs with a variable number of nodes, representing a group of graphs in batch mode requires padding `A`, `X`, and `E` to a fixed dimension.    
In order to avoid this issue, a common approach is to represent a batch of graphs with their disjoint union, leading us back to single mode.

The disjoint union of a batch of graphs is a graph where: 

1. `A` is a block diagonal matrix, constructed from the adjacency matrices of the batch;
2. `X` is obtained by stacking the node attributes of the batch;
3. `E` is a block diagonal tensor of rank 3, obtained from the edge attributes;

In order to keep track of different graphs in the disjoint union, we use an additional array of integers `I`, that maps each node to a graph with a progressive zero-based index (color coded in the image above).

Utilities for creating the disjoint union of a list of graphs are provided in `spektral.utils.data`:

```py
In [1]: from spektral.utils.data import Batch                                                               
Using TensorFlow backend.

In [2]: A_list = [np.ones((2, 2))] * 3                                                                

In [3]: X_list = [np.random.normal(size=(2, 4))] * 3                                                  

In [4]: b = Batch(A_list, X_list)                                                                     

In [5]: b.A.todense()                                                                                 
Out[5]: 
matrix([[1., 1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0.],
        [0., 0., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 1., 1.]])

In [6]: b.X                                                                                           
Out[6]: 
array([[-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
       [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
       [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
       [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
       [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
       [ 0.14221143, -0.76473164, -1.05635638,  1.45961459]])

In [7]: b.I                                                                                           
Out[7]: array([0, 0, 1, 1, 2, 2])

In [8]: b.get('AXI')                                                                                  
Out[8]: 
(<6x6 sparse matrix of type '<class 'numpy.float64'>'
  with 12 stored elements in COOrdinate format>,
 array([[-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
        [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
        [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
        [ 0.14221143, -0.76473164, -1.05635638,  1.45961459],
        [-0.85196709, -1.66795384, -1.14046868,  0.4735151 ],
        [ 0.14221143, -0.76473164, -1.05635638,  1.45961459]]),
 array([0, 0, 1, 1, 2, 2]))
```

Convolutional layers that work in single mode will work for this type of data representation, without any modification.  
Pooling layers, on the other hand, require the index vector `I` to know which nodes to pool together. 

Global pooling layers will consume `I` and reduce the graphs to single vectors. Standard pooling layers will return a reduced version of `I` along with the reduced graphs. 

---

## Conversion methods

To provide better compatibility with other libraries, Spektral has methods to convert graphs between the matrix representation (`'numpy'`) and other formats. 

The `'networkx'` format represents graphs using the Networkx library, which can then be used to [convert the graphs](https://networkx.github.io/documentation/networkx-1.10/reference/convert.html) to other formats like `.dot` and edge lists. 
Conversion utils between `'numpy'` and `'networkx'` are provided in `spektral.utils.conversion`.

---

## Molecules

When working with molecules, some specific formats can be used to represent the graphs. 

The `'sdf'` format is an internal representation format used to store an SDF file as a dictionary. A molecule in `'sdf'` format will look like this: 

```py
{'atoms': [{'atomic_num': 7,
           'charge': 0,
           'coords': array([-0.0299,  1.2183,  0.2994]),
           'index': 0,
           'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          'iso': 0},
          ...,
          {'atomic_num': 1,
           'charge': 0,
           'coords': array([ 0.6896, -2.3002, -0.1042]),
           'index': 14,
           'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
           'iso': 0}],
'bonds': [{'end_atom': 13,
           'info': array([0, 0, 0]),
           'start_atom': 4,
           'stereo': 0,
           'type': 1},
          ...,
          {'end_atom': 8,
           'info': array([0, 0, 0]),
           'start_atom': 7,
           'stereo': 0,
           'type': 3}],
'comment': '',
'data': [''],
'details': '-OEChem-03231823253D',
'n_atoms': 15,
'n_bonds': 15,
'name': 'gdb_54964',
'properties': []}
```

The `'rdkit'` format uses the [RDKit](http://www.rdkit.org/docs/index.html) library to represent molecules, and offers several methods to manipulate molecules with a chemistry-oriented approach.

The `'smiles'` format represents molecules as strings, and can be used as a space-efficient way to store molecules or perform quick checks on a dataset (e.g., counting the unique number of molecules in a dataset is quicker if all molecules are converted to SMILES first).

The `spektral.chem` and `spektral.utils` modules offer conversion methods between all of these formats, although some conversions may need more than one step (e.g., `'sdf'` to `'networkx'` to `'numpy'` to `'smiles'`). 
