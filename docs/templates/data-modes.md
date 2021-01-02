# Data modes

Creating mini-batches of data can be tricky when the samples have different shapes. 

In traditional neural networks, we're used to stretching, cropping, or padding our data so that all inputs to our models are standardized. 
For instance, images of different sizes can be modified so that they fit into a tensor of shape `[batch, width, height, channels]`.
Sequences can be padded so that they have shape `[batch, time, channels]`. And so on...

With graphs, it's a bit different. 

For instance, it is not that easy to define the meaning of "cropping" or "stretching" a graph, since these are all transformations that assume a "spatial closeness" of the pixels (which we don't have for graphs in general).

Also, it's not always the case that we have many graphs in our datasets. Sometimes, we're just interested in classifying the nodes of one big graph. Sometimes, we may have one big graph but many instances of its node features (the classification of images is one such case: one grid, many instances of pixels values). 

To make Spektral work in all of these cases, and to account for the difficulties in dealing with graphs of different sizes, we introduce the concept of **data modes**.

In Spektral, there are four of them:

- In **single mode**, we have only one graph. Node classification tasks are usually in this mode. 
- In **disjoint mode**, we represent a batch of graphs with their disjoint union. This gives us one big graph, similar to single mode, although with some differences (see below).
- In **batch mode**, we zero-pad the graphs so that we can fit them into **dense** tensors of shape `[batch, nodes, ...]`. This can be more expensive, but makes it easier to interface with traditional NNs. 
- In **mixed mode**, we have one adjacency matrix shared by many graphs. We keep the adjacency matrix in single mode (for performance, no need to duplicate it for each graph), and the node attributes in batch mode. 

In all data modes, our goal is to represent one or more graphs by grouping their respective `x`, `a` and `e` matrices into single tensors `X`, `A`, and `E`. The shapes of these tensors in the different data modes are summarized in the table below. 

|Mode      | `A.shape` | `X.shape` | `E.shape` |
|:---------|:--|:--|:--|
|`Single`  |`[nodes, nodes]`|`[nodes, n_feat]`|`[edges, e_feat]`|
|`Disjoint`|`[nodes, nodes]`|`[nodes, n_feat]`|`[edges, e_feat]`|
|`Batch`   |`[batch, nodes, nodes]`|`[batch, nodes, nodes]`|`[batch, nodes, nodes, e_feat]`|
|`Mixed`   |`[nodes, nodes]`|`[batch, nodes, n_feat]`| `[batch, edges, e_feat]` |

In the following sections we describe the four modes more into detail.
In particular, we go over which [data `Loader`](/loaders/) to use in each case.

## Single mode

<img src="https://danielegrattarola.github.io/spektral/img/single_mode.svg" style="max-width: 400px; width: 100%;"/>

In single mode we have only one graph in which: 

- `A` is a matrix of shape `[nodes, nodes]`;
- `X` is a matrix of shape `[nodes, n_feat]`;
- `E` has shape `[edges, e_feat]` so that `E[i]` corresponds to the edge in `A[i // nodes, i % nodes]`.

A very common benchmark dataset in single mode is the Cora citation network. 
We can load it with:

```py
>>> from spektral.datasets import Cora
>>> dataset = Cora()
>>> dataset
Cora(n_graphs=1)
```

As expected, we have only one graph: 

```py
>>> dataset[0]
Graph(n_nodes=2708, n_node_features=1433, n_edge_features=None, n_labels=7)

>>> dataset[0].a.shape
(2708, 2708)

>>> dataset[0].x.shape
(2708, 1433)
```

When training a GNN in single mode, we can use a `SingleLoader` that will extract the characteristic matrices from the graph and return a `tf.data.Dataset` to feed to our model:

```py
>>> from spektral.data import SingleLoader
>>> loader = SingleLoader(dataset)
>>> loader.load()
<RepeatDataset shapes: (((2708, 1433), (2708, 2708)), (2708, 7)), types: ((tf.float32, tf.int64), tf.int32)>
```

## Disjoint mode

<img src="https://danielegrattarola.github.io/spektral/img/disjoint_mode.svg" style="max-width: 400px; width: 100%;"/>

In disjoint mode we represent a set of graphs as a single graph, their "disjoint union", where:

- `A` is a sparse block diagonal matrix where each block is the adjacency matrix `a_i` of the i-th graph.
- `X` is obtained by stacking the node attributes `x_i`;
- `E` is also obtained by stacking the edges `e_i`.

The shapes of the three matrices are the same as single mode, but `nodes` is the number of all the nodes in the set of graphs. 

To keep track of the different graphs in the disjoint union, we use an additional array of zero-based indices `I` that identify which nodes belong to which graph. 
For instance: if node 8 belongs to the third graph, we will have `I[8] == 2`. <br>
In the example above, color blue represents 0, green is 1, and orange is 2

In convolutional layers, disjoint mode is indistinguishable from single mode because it is not possible to exchange messages between the disjoint components of the graph, so `I` is not needed to compute the output.  
Pooling layers, on the other hand, require `I` to know which nodes can be pooled together.


Let's load a dataset with many small graphs and have a look at the first three:

```py
>>> from spektral.datasets import TUDataset
>>> dataset = TUDataset('PROTEINS')
Successfully loaded PROTEINS.

>>> dataset = dataset[:3]
>>> dataset[0]
Graph(n_nodes=42, n_node_features=4, n_edge_features=None, n_labels=2)

>>> dataset[1]
Graph(n_nodes=27, n_node_features=4, n_edge_features=None, n_labels=2)

>>> dataset[2]
Graph(n_nodes=10, n_node_features=4, n_edge_features=None, n_labels=2)
```

To create batches in disjoint mode, we can use a `DisjointLoader`:

```py
>>> from spektral.data import DisjointLoader
>>> loader = DisjointLoader(dataset, batch_size=3)
```

Since Loaders are effectively generators, we can inspect the first batch that the loader will compute for us by calling `__next__()`:

```py
>>> batch = loader.__next__()
>>> inputs, target = batch
>>> x, a, i = inputs
>>> x.shape
(79, 4)  # 79 == 42 + 27 + 10

>>> a.shape
(79, 79)

>>> i.shape
(79, )
```

Note that, since we don't have edge attributes in our dataset, the loader did not create the `E` matrix.



## Batch mode

<img src="https://danielegrattarola.github.io/spektral/img/batch_mode.svg" style="max-width: 400px; width: 100%;"/>

In batch mode, graphs are zero-padded so that they fit into tensors of shape `[batch, N, ...]`. 
Due to the general lack of support for sparse higher-order tensors both in Scipy and TensorFlow, `X`, `A`, and `E` will be dense tensors:

- `A` has shape `[batch, nodes, nodes]`;
- `X` has shape `[batch, nodes, n_feat]`;
- `E` has shape `[batch, nodes, nodes, e_feat]` (the attributes of non-existing edges are all zeros).

If the graphs have a variable number of nodes, `nodes` will be the size of the biggest graph in the batch.

If you don't want to zero-pad the graphs or work with dense inputs, it is better to use [disjoint mode](#disjoint-mode) instead.

However, note that some pooling layers like `DiffPool` and `MinCutPool` will only work in batch mode. 

Let's re-use the dataset from the example above. We can use a `BatchLoader` as follows: 

```py
>>> from spektral.data import BatchLoader
>>> loader = BatchLoader(dataset, batch_size=3)
>>> inputs, target = loader.__next__()

>>> inputs[0].shape
(3, 42, 4)

>>> inputs[1].shape
(3, 42, 42)
```

In this case, the loader only created two inputs because we don't need the indices `I`. 
Also note that the batch was padded so that all graphs have 42 nodes, which is the size of the biggest graph out of the three.

The `BatchLoader` zero-pads each batch independently of the others, so that we don't waste memory. If you want to remove the overhead of padding each batch, you can use a `PackedBatchLoader` which will pre-pad all graphs before yielding the batches. Of course, this means that all graphs will have the same number of nodes as the biggest graph in the dataset (and not just in the batch).


## Mixed mode

<img src="https://danielegrattarola.github.io/spektral/img/mixed_mode.svg" style="max-width: 400px; width: 100%;"/>

In mixed mode we have a single graph that acts as the support for different node attributes (also sometimes called "graph signals").

In this case we have that: 

- `A` is a matrix of shape `[nodes, nodes]`;
- `X` is a tensor in batch mode, of shape `[batch, nodes, n_feat]`;
- `E` has shape `[batch, edges, e_feat]` so that `E[i, j]` corresponds to the edge of the i-th graph associated with `A[j // nodes, j % nodes]`.

Currently, there are no layers in Spektral that support mixed mode and edge attributes. 

An example of a mixed mode dataset is the MNIST random grid ([Defferrard et al., 2016](https://arxiv.org/abs/1606.09375)):

```py
>>> from spektral.datasets import MNIST
>>> dataset = MNIST()
>>> dataset
MNIST(n_graphs=70000)
```

Mixed-mode datasets have a special `a` attribute that stores the adjacency matrix, while the proper graphs that make up the dataset only have node/edge features:

```py
>>>dataset.a
<784x784 sparse matrix of type '<class 'numpy.float64'>'
    with 6396 stored elements in Compressed Sparse Row format>

>>> dataset[0]
Graph(n_nodes=784, n_node_features=1, n_edge_features=None, n_labels=1)

>>>dataset[0].a
# None
```

We can use a `MixedLoader` to deal with sharing the adjacency matrix between the graphs in our dataset: 

```py
>>> from spektral.data import MixedLoader
>>> loader = MixedLoader(dataset, batch_size=3)
>>> inputs, target = loader.__next__()

>>> inputs[0].shape
(3, 784, 1)

>>> inputs[1].shape  # Only one adjacency matrix
(784, 784)
```

Mixed mode requires a bit more work than the other three modes. In particular, it is not possible to use `loader.load()` to train a model in this mode. 

Have a look at [this example](https://github.com/danielegrattarola/spektral/blob/master/examples/other/graph_signal_classification_mnist.py) to see how to train a GNN in mixed mode.
