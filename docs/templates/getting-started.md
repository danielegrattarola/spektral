# Getting started

Spektral is designed according to the guiding principles of Keras to make things extremely simple for beginners while maintaining flexibility for experts.  

In this tutorial, we will go over the main features of Spektral while creating a graph neural network for graph classification. 

## Graphs

A graph is a mathematical object that represents relations between entities. We call the entities "nodes" and the relations "edges". 

Both the nodes and the edges can have vector **features**.

In Spektral, graphs are represented with instances of `spektral.data.Graph`.
A graph can have four main attributes: 

- `a`: the **adjacency matrix**
- `x`: the **node features**
- `e`: the **edge features**
- `y`: the **labels**

A graph can have all of these attributes or none of them. Since Graphs are just plain Python objects, you can also add extra attributes if you want. For instance, see `graph.n_nodes`, `graph.n_node_features`, etc.

### Adjacency matrix (`graph.a`)

Each entry `a[i, j]` of the adjacency matrix is non-zero if there exists an edge going from node `j` to node `i`, and zero otherwise. 

We can represent `a` as a dense `np.array` or as a Scipy sparse matrix of shape `[n_nodes, n_nodes]`.
Using an `np.array` to represent the adjacency matrix can be expensive, since we need to store a lot of 0s in memory, so sparse matrices are usually preferable.

With sparse matrices, we only need to store the non-zero entries of `a`. In practice, we can implement a sparse matrix by only storing the indices and values of the non-zero entries in a list, and assuming that if a pair of indices is missing from the list then its corresponding value will be 0.  
This is called the _COOrdinate format_ and it is the format used by TensorFlow to represent sparse tensors.

For example, the adjacency matrix of a weighted ring graph with 4 nodes:

```
[[0, 1, 0, 2],
 [3, 0, 4, 0],
 [0, 5, 0, 6],
 [7, 0, 8, 0]]
```

can be represented in COOrdinate format as follows: 

```python
R, C, V
0, 1, 1
0, 3, 2
1, 0, 3
1, 2, 4
2, 1, 5
2, 3, 6
3, 0, 7
3, 2, 8
```

where `R` indicates the "row" indices, `C` the columns, and `V` the non-zero values `a[i, j]`. For example, in the second line, we see that there is an edge that goes **from node 3 to node 0** with weight 2.

We also see that, in this case, all edges have a corresponding edge that goes in the opposite direction. For the sake of this example, all edges have been assigned a different weight. In practice, however, edge `i, j` will often have the same weight as edge `j, i` and the adjacency matrix will be symmetric.

Many convolutional and pooling layers in Spektral use this sparse representation of matrices to do their computation, and sometimes you will see in the documentation a comment saying that **"This layer expects a sparse adjacency matrix."**

### Node features (`graph.x`)

When working with graph neural networks, we usually associate a vector of features with each node of a graph. This is no different from how every pixel in an image has an `[R, G, B, A]` vector associated with it. 

Since we have `n_nodes` nodes and each node has a feature vector of size `n_node_features`, we can stack all features in a matrix `x` of shape `[n_nodes, n_node_features]`.

In Spektral, `x` is always represented with a dense `np.array` (since in this case we don't run the risk of storing many useless zeros -- at least not often).

### Edge features (`graph.e`)

Similar to node features, we can also have features associated with edges. These are usually different from the _edge weights_ that we saw for the adjacency matrix, and often represent the kind of relation between two nodes (e.g., acquaintances, friends, or partners).

When representing edge features, we run into the same problems that we have for the adjacency matrix. 

If we store them in a dense `np.array`, then the array will have shape `[n_nodes, n_nodes, n_edge_features]` and most of its entries will be zeros. Unfortunately, order-3 tensors cannot be represented as Scipy sparse matrices, so we need to be smart about it. 

Similar to how we stored the adjacency matrix as a list of entries `r, c, v`, here we can use the COOrdinate format to represent our edge features. 
Assume that, in the example above, each edge has `n_edge_features=3` features. We could do something like: 

```python
R, C, V
0, 1, [ef_1, ef_2, ef_3]
0, 3, [ef_1, ef_2, ef_3]
1, 0, [ef_1, ef_2, ef_3]
1, 2, [ef_1, ef_2, ef_3]
2, 1, [ef_1, ef_2, ef_3]
2, 3, [ef_1, ef_2, ef_3]
3, 0, [ef_1, ef_2, ef_3]
3, 2, [ef_1, ef_2, ef_3]
```

Since we already have the information of `R` and `C` in the adjacency matrix, we only need to store the `V` column as a matrix `e` of shape `[n_edges, n_edge_features]`. In this case, `n_edges` indicates the number of non-zero entries in the adjacency matrix. 

Note that, since we have separated the edge features from the edge indices of the adjacency matrix, the order in which we store the edge features is very important. We must not break the correspondence between the edges in `a` and the edges in `e`.

**In Spektral, we always assume that edges are sorted in the row-major ordering (we first sort by row, then by column, like in the example above). This is not important when building the adjacency matrix, but it is important when building `e`.**

You can use `spektral.utils.sparse.reorder` to sort a matrix of edge features in the correct row-major order given by an _edge index_ (i.e., the matrix obtained by stacking the `R` and `C` columns).

### Labels (`graph.y`)

Finally, in many machine learning tasks we want to predict a label given an input. 
When working with GNNs, labels can be of two types: 

1. **Graph labels** represent some global properties of an entire graph;
2. **Node labels** represent some properties of each individual node in a graph;

Spektral supports both kinds. 

Labels are dense `np.array`s or scalars, stored in the `y` attribute of a `Graph` object.    
Graph-level labels can be either scalars or 1-dimensional arrays of shape `[n_labels, ]`.   
Node-level labels can be 1-dimensional arrays of shape `[n_nodes, ]` (representing a scalar label for each node), or 2-dimensional arrays of shape `[n_nodes, n_labels]`.

This difference is relevant only when using a [`DisjointLoader`](/loaders/#disjointloader) ([read more here](/data-modes/#disjoint-mode)).

## Datasets

The `spektral.data.Dataset` container provides some useful functionality to manipulate collections of graphs.

Let's load a popular benchmark dataset for graph classification: 

```python
>>> from spektral.datasets import TUDataset

>>> dataset = TUDataset('PROTEINS')

>>> dataset
TUDataset(n_graphs=1113)
```

We can now retrieve individual graphs:

```python
>>> dataset[0]
Graph(n_nodes=42, n_node_features=4, n_edge_features=None, y=[1. 0.])
```

or shuffle the data:

```python
>>> np.random.shuffle(dataset)
```

or slice the dataset into sub-datsets: 

```python
>>> dataset[:100]
TUDataset(n_graphs=100)
```

Datasets also provide methods for applying **transforms** to each datum: 

- `apply(transform)` - modifies the dataset in-place, by applying the `transform` to each graph;
- `map(transform)` - returns a list obtained by applying the `transform` to each graph;
- `filter(function)` - removes from the dataset any graph for which `function(graph)` is `False`. This is also an in-place operation.

For example, let's modify our dataset so that we only have graphs with less than 500 nodes:

```python
>>> dataset.filter(lambda g: g.n_nodes < 500)

>>> dataset
TUDataset(n_graphs=1111)  # removed 2 graphs
``` 

Now let's apply some transforms to our graphs. For example, we can modify each graph so that the node features also contain the one-hot-encoded degree of the nodes.

First, we compute the maximum degree of the dataset, so that we know the size of the one-hot vectors: 

```python
>>> max_degree = dataset.map(lambda g: g.a.sum(-1).max(), reduce=max)

>>> max_degree
12
```

Try to go over the lambda function to see what it does. Also, notice that we passed a reduction function to the method, using the `reduce` keyword. This will be run on the output list computed by the map.

Now we are ready to augment our node features with the one-hot-encoded degree. Spektral has a lot of pre-implemented `transforms` that we can use: 

```python
>>> from spektral.transforms import Degree

>>> dataset.apply(Degree(max_degree))
```

We can see that it worked because now we have an extra `max_degree + 1` node features:

```python
>>> dataset[0]
Graph(n_nodes=42, n_node_features=17, n_edge_features=None, y=[1. 0.])
```

Since we will be using a `GCNConv` layer in our GNN, we also want to follow the [original paper](https://arxiv.org/abs/1609.02907) that introduced this layer and do some extra pre-processing of the adjacency matrix. 

Since this is a fairly common operation, Spektral has a transform to do it: 

```python
>>> from spektral.transforms import GCNFilter

>>> dataset.apply(GCNFilter())
```

Many layers will require you to do some form of preprocessing. If you don't want to go back to the literature every time, every convolutional layer in Spektral has a `preprocess(a)` method that you can use to transform the adjacency matrix as needed. <br>
Have a look at the handy [`LayerPreprocess` transform](/transforms/#layerpreprocess).


## Creating a GNN

Creating GNNs is where Spektral really shines. Since Spektral is designed as an extension of Keras, you can plug any Spektral layer into a Keras `Model` without modifications.  
We just need to use the functional API because GNN layers usually need two or more inputs (so no `Sequential` models for now). 

For our first GNN, we will create a simple network that first does a bit of graph convolution, then sums all the nodes together (known as "global pooling"), and finally classifies the result with a dense softmax layer. We will also use dropout for regularization.

Let's start by importing the necessary layers:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
```

Now we can use model subclassing to define our model:

```python
class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out
``` 

And that's it!

Note how we mixed layers from Spektral and Keras interchangeably: it's all just computation with tensors underneath.

This also means that if you want to break free from `Graph` and `Dataset` and every other feature of Spektral, you can. 

**Note:** If you don't want to subclass `Model` to implement your GNN, you can also use the classical declarative style. You just need to pay attention to the `Input` and leave "node" dimensions unspecified (so `None` instead of `n_nodes`).


## Training the GNN

Now we're ready to train the GNN. First, we instantiate and compile our model: 

```python
model = MyFirstGNN(32, dataset.n_labels)
model.compile('adam', 'categorical_crossentropy')
```

and we're almost there!

However, here's where graphs get in our way. Unlike regular data, like images or sequences, graphs cannot be stretched, cut, or reshaped so that we can fit them into tensors of pre-defined shapes. If a graph has 10 nodes and another one has 4, we have to keep them that way. 

This means that iterating over a dataset in mini-batches is not trivial and we cannot simply use the `model.fit()` method of Keras as-is. 

We have to use a data `Loader`.

### Loaders

Loaders iterate over a graph dataset to create mini-batches. They hide a lot of the complexity behind the process so that you don't need to think about it. 
You only need to go to [this page](/data-modes) and read up on **data modes**, so that you know which loader to use. 

Each loader has a `load()` method that returns a data generator that Keras can process. 

Since we're doing graph-level classification, we can use a `BatchLoader`. It's a bit slow and memory intensive (a `DisjointLoader` would have been better), but it lets us simplify the definition of `MyFirstGNN`. Again, go read about [data modes](/data-modes) after this tutorial.

Let's create a data loader:

```python
from spektral.data import BatchLoader

loader = BatchLoader(dataset_train, batch_size=32)
```

and we can finally train our GNN!

Since loaders are essentially generators, we need to provide the `steps_per_epoch` keyword to `model.fit()` and we don't need to specify a batch size:

```python
model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=10)
``` 

Done!

## Evaluating the GNN

Evaluating the performance of our model, be it for testing or validation, follows a similar workflow. 

We create a data loader: 

```python
from spektral.data import BatchLoader

loader = BatchLoader(dataset_test, batch_size=32)
```

and feed it to the model by calling `load()`:

```python
loss = model.evaluate(loader.load(), steps=loader.steps_per_epoch)

print('Test loss: {}'.format(loss))
```

## Node-level learning

Besides learning to predict labels for the whole graph, like in this tutorial, GNNs are very effective at learning to predict labels for each node. This is called "node-level learning" and we usually do it for datasets with one big graph (think a social network).

For example, reproducing the results of the [GCN paper for classifying nodes in a citation network](https://arxiv.org/abs/1609.02907) can be done with `GCNConv` layers, the `Citation` dataset, and a `SingleLoader`: check out [this example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py).

As a matter of fact, check out [all the examples](/examples).

## Go create!

You are now ready to use Spektral to create your own GNNs. 

If you want to build a GNN for a specific task, chances are that everything you need is already in Spektral. Check out the [examples](https://github.com/danielegrattarola/spektral/tree/master/examples) for some ideas and practical tips.

Remember to read the [data modes section](/data-modes) to learn about representing graphs and creating mini-batches. 

Make sure to read the documentation, and get in touch [on Github](https://github.com/danielegrattarola/spektral) if you have a feature that you want to see implemented. 

If you want to cite Spektral in your work, refer to our paper: 

> [Graph Neural Networks in TensorFlow and Keras with Spektral](https://arxiv.org/abs/2006.12138) <br>
> Daniele Grattarola and Cesare Alippi  

