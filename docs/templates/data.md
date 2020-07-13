# Representing graphs

In Spektral, graphs are represented as matrices:

- `A` is the adjacency matrix of shape `(N, N)`, where `N` is the number of nodes. `A` is a binary matrix where `A[i, j] = 1` if there is an edge between nodes `i` and `j`, and `0` otherwise. 
- `X` is the node attributes matrix of shape `(N, F)`, where `F` is the size of the node attributes. 

Sometimes, we can also have edge attributes of size `S`, which we store in a matrix `E` of shape `(n_edges, S)` where each row is associated to a non-zero entry of `A`: assuming that `A` is a Scipy sparse matrix, we have that `E[i]` is the attribute associated to `A.data[i]`.

## Modes

Spektral supports four different ways of representing graphs or batches of graphs, which we refer to as **data modes**.

- In **single mode**, we have one graph with its adjacency matrix and attributes;
- **Disjoint mode** is a special case of single mode, where the graph is the disjoint union of a set of graphs;
- In **batch mode**, a set of graphs is represented by stacking their adjacency and node attributes matrices in higher order tensors of shape `(batch, N, ...)`;
- In **mixed mode**, we have a single adjacency matrix shared by a set of graphs; the adjacency matrix will be in single mode, but the node attributes will be in batch mode. 

The difference between the four data modes can be easily seen in how `A`, `X`, and `E` have different shapes in each case:

|Mode    | `A.shape`     | `X.shape`     | `E.shape`        |
|:------:|:-------------:|:-------------:|:----------------:|
|Single  |`(N, N)`       |`(N, F)`       |`(n_edges, S)`    |
|Disjoint|`(N, N)`       |`(N, F)`       |`(n_edges, S)`    |
|Batch   |`(batch, N, N)`|`(batch, N, F)`|`(batch, N, N, S)`|
|Mixed   |`(N, N)`       |`(batch, N, F)`| N/A              |



## Single mode

<img src="https://danielegrattarola.github.io/spektral/img/single_mode.svg" width="50%"/>

In **single mode** the data describes a single graph where:

- `A` is a sparse matrix of shape `(N, N)`;
- `X` is a matrix of shape `(N, F)`;

When edge attributes are present, we represent them as a matrix `E` of shape `(n_edges, S)` so that there is a correspondence between `E[i]` and `A.data[i]`.

Three very popular datasets in this setting are the citation networks: Cora, Citeseer, and Pubmed. To load a citation network, you can use the built-in loader:

```py
>>> from spektral.datasets import citation
>>> A, X, _, _, _, _ = citation.load_data('cora')
Loading cora dataset
>>> A.shape
(2708, 2708)
>>> X.shape
(2708, 1433)
```

## Disjoint mode

<img src="https://danielegrattarola.github.io/spektral/img/disjoint_mode.svg" width="50%"/>

**Disjoint mode** is a smart way of representing a set of graphs as a single graph.
In particular, the disjoint union of a batch is a graph where 

- `A` is a sparse block diagonal matrix, where each block is the adjacency matrix `A_i` of the i-th graph;
- `X` is obtained by stacking the node attributes matrices of the graphs.

When edge attributes are present, we represent them as a matrix `E` of shape `(n_edges, S)` so that there is a correspondence between `E[i]` and `A.data[i]`.

In order to keep track of different graphs in the disjoint union, we use an additional array of integers `I` that identifies which nodes belong to the same graph.  
For convolutional layers, disjoint mode is indistinguishable from single mode because it is not possible to exchange messages between the components of the graph, so `I` is not needed to compute the output.  
Pooling layers, on the other hand, require `I` to know which nodes can be pooled together. 
Hierarchical pooling layers will return a reduced version of `I` along with the reduced graphs. Global pooling layers will consume `I` and reduce the graphs to single vectors. 

Utilities for creating the disjoint union of a list of graphs are provided in `spektral.utils.data`:

```py
>>> from spektral.utils.data import numpy_to_disjoint
>>> A_list = [np.ones((2, 2)), np.ones((3, 3))]  # One graph has 2 nodes, the other has 3
>>> X_list = [np.random.randn(2, 4), np.random.randn(3, 4)]  # F = 4
>>> X, A, I = numpy_to_disjoint(X_list, A_list)
>>> X.shape
(5, 4)
>>> A.shape
(5, 5)
>>> A.toarray()
array([[1., 1., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [0., 0., 1., 1., 1.],
       [0., 0., 1., 1., 1.],
       [0., 0., 1., 1., 1.]])
>>> I
array([0, 0, 1, 1, 1])
```

## Batch mode

<img src="https://danielegrattarola.github.io/spektral/img/batch_mode.svg" width="50%"/>

In **batch mode**, graphs have the same number of nodes and are stacked in tensors of shape `(batch, N, ...)`. 
Due to the general lack of support for sparse higher-order tensors both in Scipy and TensorFlow, `A` and `X` will be dense tensors.

In this case, edge attributes must also be reshaped and made dense, so that `E` has shape `(batch, N, N, S)` (the attributes of non-existing edges are usually all zeros).

Note that if the graphs have variable number of nodes, the matrices must be zero-padded so that they have the same `N`.
If you don't want to zero-pad the graphs or work with dense inputs, it is better to work in [disjoint mode](https://danielegrattarola.github.io/spektral/data/#disjoint-mode) instead.

The advantage of batch mode is that it is more intuitive and it allows to use the training loop of `tf.keras` without any modifications. Also, some pooling layers like `DiffPool` and `MinCutPool` will only work in batch mode. 

For example, the QM9 dataset of small molecules will be loaded in batch mode by default:

```py
>>> from spektral.datasets import qm9
>>> A, X, E, y = qm9.load_data()
Loading QM9 dataset.
Reading SDF
>>> A.shape
(133885, 9, 9)
>>> X.shape
(133885, 9, 6)
>>> E.shape
(133885, 9, 9, 5)
```

## Mixed mode

<img src="https://danielegrattarola.github.io/spektral/img/mixed_mode.svg" width="50%"/>

In **mixed mode** we consider a single adjacency matrix that acts as the support for different node attributes (also sometimes called "signals").

In this case we have that: 

- `A` is a sparse matrix of shape `(N, N)`;
- `X` is a tensor in batch mode, of shape `(batch, N, F)`;

Currently, there are no layers in Spektral that support mixed mode and edge attributes. 

An example of a mixed mode dataset is the MNIST random grid ([Defferrard et al., 2016](https://arxiv.org/abs/1606.09375)):

```py
>>> from spektral.datasets import mnist
>>> X_tr, y_tr, X_va, y_va, X_te, y_te, A = mnist.load_data()
>>> A.shape
(784, 784)
>>> X_tr.shape
(50000, 784, 1)
```
