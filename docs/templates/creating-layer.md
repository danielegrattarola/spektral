# Creating a Message-Passing Layer

In this tutorial we go over the `MessagePassing` interface for creating GNN layers. 

This is a very flexible class that is based on three main functions: message, aggregate and update.
By overriding these methods, you can define the behaviour of your own layers. 

## Essential information

The `MessagePassing` layer can be subclassed to create layers that work in [single and disjoint mode](/data-modes) using sparse adjacency matrices. This ensures that your layers will work for both node-level and graph-level learning while being very computationally efficient.

The functionality of these layers is defined by the `message`, `aggregate` and `update` methods, and is summarized as follows: 

```python

x_out[i] = update(x[i], aggregate([message(x[j]) for j in neighbours(i)]))

```

The `message` function computes a transformation of the neighbours of each node. The `aggregate` function aggregates the messages in a way that is independent of the order in which the messages are processed (like a sum, an average, the maximum, etc). The `update` function takes the aggregated messages from the neighbours and decides how to transform each node.

This message-passing scheme is computed by calling the `propagate` method of the class, which will return the updated node features (`x_out`).

## Example

In this example we will implement a graph convolutional network ([Kipf & and Welling, 2016](https://arxiv.org/abs/1609.02907)) using the MessagePassing interface. 

First, let's add some trainable parameters when creating the layer: 

```py
class GCN(MessagePassing):
    def __init__(self, n_out, activation):
        super().__init__(activation=activation)
        self.n_out = n_out

    def build(self, input_shape):
        n_in = input_shape[0][-1]
        self.weights = self.add_weight(shape=(n_in, self.n_out))
```

Note that the Keras keyword `activation` was passed to the constructor of the superclass. This can be done with any Keras keyword (like regularizers, constraints, etc) and the layer will process them automatically. 

By default, the `call` method of MessagePassing layers will only call `propagate`. We modify it so that it also transforms the node features before starting the propagation:

```py
def call(self, inputs):
    x, a = inputs

    # Update node features
    x = tf.matmul(x, self.weights)

    return self.propagate(x=x, a=a)
```

Then, we implement the `message` function.
The `get_i` and `get_j` built-in methods can be used to automatically access either side of the edges \(i \leftarrow j\). For instance, we can use `get_j` to access the node features `x[j]` of all neighbors `j`.

If you need direct access to the edge indices, you can use the `index_i` and `index_j` attributes.

In this case, we only need to get the neighbors' features and return them: 

```py
def message(self, x):
    # Get the node features of all neighbors
    return self.get_j(x)
```

Then, we define an aggregation function for the messages. We can use a simple average of the nodes:

```py
from spektral.layers.ops import scatter_mean

def aggregate(self, messages):
    return scatter_mean(messages, self.index_i, self.n_nodes)
```

**Note**: `n_nodes` is computed dynamically at the start of propagation, exactly like `index_i`.

Since there are a few common aggregation functions that are often used in the literature, you can also skip the implementation of this method and simply pass a special keyword to the `__init__()` method of the superclass:

```py
def __init__(self):
    # Equivalent to the above implementation of aggregate
    super().__init__(aggregate='mean')
```

Finally, we can use the `update` method to apply the activation function: 

```py
def update(self, embeddings):
    return self.activation(embeddings)
```

This is enough to get started with building your own layers in Spektral. 

## Notes

An important feature of the MessagePassing class is that any extra keyword argument given to `propagate`, will be compared to the signatures of `message`, `aggregate` and `update` and forwarded to those functions if a match is found. 

For example, we can call:

```py
propagate(x=x, a=a, extra_tensor=extra_tensor)
```

and define the message function as: 

```py
def message(self, x, extra_tensor=None):
    ...  # Do something with extra_tensor
```


Finally, we already noted that MessagePassing layers only support single and disjoint mode, and they also require that the adjacency matrix is a SparseTensor. 

If you need more control on your layers, you can have a look at `spektral.layers.Conv` for a stripped-down class that performs no checks on the inputs and only implements some essential features like keyword parsing. 

For example, `spektral.layers.GCNConv` implements the same GCN layer that we just saw, using the `Conv` class so that it can provide support for batch and mixed mode, as well as dense adjacency matrices. 
