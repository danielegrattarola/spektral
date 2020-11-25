# Getting started

Spektral is designed according to the guiding principles of the Keras API to make things extremely simple for beginners while maintaining flexibility for experts.  

In this page we will go over the main features of Spektral while creating a graph neural network for graph classification. 

## Graphs

A graph is a mathematical object that represents relations between objects. We call the objects "nodes" and the relations "edges". 

Both the nodes and the edges can have vector **features**.

In Spektral, graphs are represented with instances of `spektral.data.Graph` which can contain:

- `a`: the **adjacency matrix** - usually a `scipy.sparse` matrix of shape `(n_nodes, n_nodes)`. 
- `x`: the **node features** - represented by a `np.array` of shape `(n_nodes, n_node_features)`.
- `e`: the **edge features** - usually represented in a sparse edge list format, with a `np.array` of shape `(n_edges, n_edge_features)`.
- `y`: the **labels** - can represent anything, from graph labels to node labels, or even something else. 

A graph can have all of these attributes or none of them. You can even add extra attributes if you want: after all, a `Graph` is just a plain Python object. For instance, see `graph.n_nodes`, `graph.n_node_features`, etc.

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

or slice the dataset up into sub-datsets: 

```python
>>> dataset[:100]
TUDataset(n_graphs=100)
```

Datasets also provide methods for applying **transforms** to each data: 

- `apply(transform)` - modifies the dataset in-place, by applying the `transform` to each graph;
- `map(transform)` - returns a list obtained by applying the `transform` to each graph;
- `filter(function)` - removes from the dataset any graph for which `function(graph)` is `False`. This is also an in-place operation.

For exampe, let's modify our dataset so that we only have graphs with less than 500 nodes:

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

Try to go over the lambda function to see what it does. Also, notice that we passed another function to the `reduce` keyword. Can you guess why?

Now we are ready to augment our node features with the one-hot-encoded degree. Spektral has a lot of pre-implemented `transforms` that we can use: 

```python
>>> from spektral.transforms import Degree

>>> dataset.apply(Degree(max_degree))
```

We can see that it worked because now we have and extra `max_degree + 1` node features, which are our one-hot vectors:

```python
>>> dataset[0]
Graph(n_nodes=42, n_node_features=17, n_edge_features=None, y=[1. 0.])
```

Since we will be using a `GraphConv` layer in our GNN, we also want to follow the [original paper](https://arxiv.org/abs/1609.02907) that introduced this layer, and do some extra pre-processing. 

Specifically, we need to normalize the adjacency matrix of each graph by the node degrees. Since this is a fairly common operation, Spektral has a transform to do it: 

```python
>>> from spektral.transforms import GCNFilter

>>> dataset.apply(GCNFilter())
```

Many layers will require you to do some form of preprocessing. If you don't want to go back to the literature every time, you can use the handy [`LayerPreprocess` transform](/transforms/#layerpreprocess).


## Creating a GNN

Creating GNNs is where Spektral really shines. Since Spektral is designed as an extension of Keras, you can plug any Spektral layer into a Keras `Model` without modifications.  
We just need to use the functional API because GNN layers usually need two or more inputs (so no `Sequential` models for now). 

For our first GNN, we will create a simple network that first does a bit of graph convolution, then sums all the nodes together (known as "global pooling"), and finally classifies the result with a dense softmax layer.  
Oh, and we will also use dropout for regularization.

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
        self.graph_conv = GraphConv(n_hidden)
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

And that's it.

Note how we mixed layers from Spektral and Keras interchangeably: it's all just computation with tensors underneath! 

This also means that if you want to break free from `Graph` and `Dataset` and every other feature of Spektral, you can. 

**Note:** If you don't want to subclass `Model` to implement your GNN, you can also use the classical declarative style. You just need to pay attention to the `Input` and leave "node" dimensions unspecified (so `None` instead of `n_nodes`).


## Training the GNN

Now we're ready to train the GNN. First, we instantiate and compile our model: 

```python
model = MyFirstGNN(32, dataset.n_labels)
model.compile('adam', 'categorical_crossentropy')
```

and we're almost there!

However, here's where graphs get in our way. Unlike regular data, like images or sequences, graphs cannot be stretched or cut or reshaped so that we can fit them into tensors of pre-defined shape. If a graph has 10 nodes and another one has 4, we have to keep them that way. 

This means that iterating over a dataset in mini-batches is not trivial and we cannot simply use the `model.fit()` method of Keras as-is. 

We have to use a data `Loader`.

### Loaders

Loaders iterate over a graph dataset to create mini-batches. They hide a lot of the complexity behind the process, so that you don't need to think about it. 
You only need to go to [this page](/data-modes) and read up on **data modes**, so that you know which loader to use. 

Each loader has a `load()` method that when called will return a data generator that Keras can process. 

Since we're doing graph-level classification, we can use a `BatchLoader`. It's a bit slow and memory intensive (a `DisjointLoader` would have been better), but it lets us simplify the definition of `MyFirstGNN`. Again, go read about [data modes](/data-modes) after this tutorial.

Let's create a data loader:

```python
from spektral.data import BatchLoader

loader = BatchLoader(dataset_train, batch_size=32)
```

and we can finally train our GNN!

Since loaders are essentially generators, we need to provide the `steps_per_epoch` keyword to `model.fit()` and we don't need to specify a batch size:

```python
model.fit(loader.load(),
          steps_per_epoch=loader.steps_per_epoch,
          epochs=10)
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
loss = model.evaluate(loader.load(), 
                      steps=loader.steps_per_epoch)
print('Test loss: {}'.format(loss))
```

## Node-level learning

Besides learning to predict labels for the whole graph, like in this tutorial, GNNs are very effective at learning to predict labels for each individual node. This is called "node-level learning" and we usually do it for datasets with one big graph (think a social network).

For example, reproducing the results of the [GCN paper for classifying nodes in a citation network](https://arxiv.org/abs/1609.02907) can be done with `GraphConv` layers, the `Citation` dataset, and a `SingleLoader`: check out [this example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py).

As a matter of fact, check out [all the examples](/examples).

## Go create!

You are now ready to use Spektral to create your own GNNs. 

If you want to build a GNN for a specific task, chances are that everything you need is already in Spektral. Check out the [examples](https://github.com/danielegrattarola/spektral/tree/master/examples) for some ideas and practical tips.

Remember to read the [data modes section](/data-modes) to learn about representing graphs and creating mini-batches. 

Make sure to read the documentation, and get in touch [on Github](https://github.com/danielegrattarola/spektral) if you have a feature that you want to see implemented. 

If you want to cite Spektral in your work, refer to our paper: 

> Graph Neural Networks in TensorFlow and Keras with Spektral  
> D. Grattarola and C. Alippi  
> ICML 2020 - GRL+ Workshop  
> [https://arxiv.org/abs/2006.12138](https://arxiv.org/abs/2006.12138)  
