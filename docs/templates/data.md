## Representing graphs

Spektral uses a matrix-based representation for manipulating graphs and feeding them to neural networks. This approach is one of the most commonly used in the literature on graph neural networks, and it's perfect to perform efficient computations on GPU.

A graph is generally represented by three matrices:

- \(A \in \{0, 1\}^{N \times N}\): a binary adjacency matrix, where \(A_{ij} = 1\) if there is a connection between nodes \(i\) and \(j\), and \(A_{ij} = 0\) otherwise;
- \(X \in \mathbb{R}^{N \times F}\): a matrix encoding node attributes (or features), where an \(F\)-dimensional attribute vector is associated to each node;
- \(E \in \mathbb{R}^{N \times N \times S}\): a matrix encoding edge attributes, where an \(S\)-dimensional attribute vector is associated to each edge;

Some formulations (like the [graph neural networks](https://arxiv.org/abs/1806.01261) proposed by Battaglia et al.) include a feature vector describing the global state of the graph, but this is not supported by Spektral for now. 

## Modes

In Spektral, some layers and functions are implemented to work on a single graph, while others consider sets (i.e., datasets or batches) of graphs. 

To understand the need for different settings, consider the difference between a citation network and a dataset of molecules.
In the citation network, the elements of interest are the single nodes and the connections between them. Node and edge features are usually specific to each individual network, and there is no point in training a model across networks.  
On the other hand, when working with molecules in a dataset, we are in a much more familiar setting. Each molecule can be seen as a point in some space, and learning tasks are applied to this domain. Here, the atoms and bonds that make up the molecules are repeated across the dataset with high frequency (like pixel colors in images, where information is mostly encoded in the spatial relations between pixels rather than in the colors themselves), and we are interested in finding patterns that describe the properties of the molecules in general. 

We distinguish between three main _modes_ of operation: 

- **single**, where we consider a single graph, with its topology and attributes;
- **batch**, where we consider a collection of graphs, each with its own topology and attributes;
- **mixed**, where we consider a graph with fixed topology, but a collection of 
different attributes; this can be seen as a particular case of the batch mode (i.e., the case where all adjacency matrices are the same) but is treated separately for computational reasons. 

|Mode  |Adjacency    |Node attr.   |Edge attr.      |
|:----:|:-----------:|:-----------:|:--------------:|
|Single|(N, N)       |(N, F)       |(N, N, S)       |
|Batch |(batch, N, N)|(batch, N, F)|(batch, N, N, S)|
|Mixed |(N, N)       |(batch, N, F)|(batch, N, N, S)|


In practice, Spektral assumes that when operating in "single" mode the data has no `batch` dimension, and describes a single graph:

```py
In [1]: from spektral.datasets import citation
Using TensorFlow backend.

In [2]: adj, node_features, _, _, _, _, _, _ = citation.load_data('cora')
Loading cora dataset

In [3]: adj.shape
Out[3]: (2708, 2708)

In [4]: node_features.shape
Out[4]: (2708, 1433)
```

This means that when training models in single mode, the usual batching of the data along the first dimension cannot be done (unless the graph can be safely "sliced"), and the whole graph must be fed to the model at each step. See [the semi-supervised classification example](https://github.com/danielegrattarola/spektral/blob/master/examples/semi_sup_classification_gcn.py) for a better understanding.

In "batch" mode, the matrices will have the `batch` dimension first: 

```py
In [1]: from spektral.datasets import qm9
Using TensorFlow backend.

In [2]: adj, nf, ef, _ = qm9.load_data()
Loading QM9 dataset.
Reading SDF
100%|█████████████████████| 133885/133885 [00:29<00:00, 4579.22it/s]

In [3]: adj.shape
Out[3]: (133885, 9, 9)

In [4]: nf.shape
Out[4]: (133885, 9, 6)

In [5]: ef.shape
Out[5]: (133885, 9, 9, 1)
```

which should not surprise you if you are familiar with machine learning frameworks (try and load MNIST or any other benchmark image dataset to see a similar representation of the data).

Finally, in "mixed" mode, where we consider a single adjacency matrix (with no `batch` dimension), and collections of node and edge attributes (with the `batch` dimension).

## Other formats

To provide better compatibility with other libraries, Spektral has methods to convert graphs between the matrix representation (`'numpy'`) and other formats. 

The `'networkx'` format represents graphs using the Networkx library, which can then be used to [convert the graphs](https://networkx.github.io/documentation/networkx-1.10/reference/convert.html) to other formats like `.dot` and edge lists. 
Conversion methods between `'numpy'` and `'networkx'` are provided in `spektral.utils.conversion`.


### Molecules

When working with molecules, some specific formats can be used to represent the graphs. 

The `'sdf'` format is an internal representation format used to store an SDF file to a dictionary. A molecule in `'sdf'` format will look something like this: 

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

The `spektral.chem` modules offers conversion methods between all of these formats, although some conversions may need more than one step to do (e.g., `'sdf'` to `'networkx'` to `'numpy'` to `'smiles'`). Support for direct conversion between all formats will eventually be added.