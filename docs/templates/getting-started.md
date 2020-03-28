# Getting started

Spektral is designed according to the guiding principles of the Keras API to make things extremely simple for beginners while maintaining flexibility for experts and researchers.  

The most important modules of Spektral are `layers.convolutional` and `layers.pooling`, which offer a number of popular layers to start building graph neural networks (GNNs) right away.     
Because Spektral is designed as an extension of Keras, you can plug any Spektral layer into an existing Keras `Model` without modifications. 

## Node classification on citation networks

In this example, we will build a simple [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) for semi-supervised classification of nodes.

This is a simple but challenging task that consists of classifying text documents in a **citation network**.   
In this type of graph, each node represents a document and is associated to a binary bag-of-words attribute (1 if a given word appears in the text, 0 otherwise). 
If a document cites another, then there exist an undirected edge between the two corresponding nodes. 
Finally, each node has a class label that we want to predict. 

This is a **transductive** learning setting, where we observe all of the nodes and edges at training time, but only a fraction of the labels. The goal is to learn to predict the missing labels.

The `datasets.citation` module of Spektral lets you to download and load three popular citation datasets (Cora, Citeseer and Pubmed) in one line of code. For instance, loading the Cora dataset is as simple as: 

```python
from spektral.datasets import citation
data = citation.load_data('cora')
A, X, y, train_mask, val_mask, test_mask = data
X = X.toarray()

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]
```

This will load the network's adjacency matrix `A` as a Scipy sparse matrix of shape `(N, N)`, the node features `X` of shape `(N, F)`, and the labels `y` of shape `(N, n_classes)`. The loader will also return some boolean masks to know which nodes belong to the training, validation and test sets (`train_mask, val_mask, test_mask`).


## Creating a GNN

To create a GCN, we will use the `GraphConv` layer and the functional API of Keras:

```python
from spektral.layers import GraphConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
```

Building the model is no different than building any Keras model, but we will need to provide multiple inputs to the `GraphConv` layers (namely `A` and `X`):

```python
# Model definition
X_in = Input(shape=(F, ))  # Input layer for X
A_in = Input((N, ), sparse=True)  # Input layer for A

graph_conv_1 = GraphConv(16, activation='relu')([X_in, A_in])
dropout = Dropout(0.5)(graph_conv_1)
graph_conv_2 = GraphConv(n_classes, activation='softmax')([dropout, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_conv_2)
```

And that's it. We just built our first GNN in Spektral and Keras. 

Note how we used the familiar API of Keras to create the GCN layers, as well as the standard `Dropout` layer to regularize our model. All features of Keras are also supported by Spektral (including initializers, regularizers, etc.).

An important thing to notice at this point is how we defined the `Input` layers of our model. Because the "elements" of our dataset are the node themselves, we are telling Keras to consider each node as a separate sample, so that the "batch" axis is implicitly defined as `None`.  
In other words, a sample of the node attributes will be a vector of shape `(F, )` and a sample of the adjacency matrix will be one of its rows of shape `(N, )`. 

Keep this detail in mind for later. 


## Training the GNN

Before training the model, we have to pre-process the adjacency matrix to scale the weights of a node's connections according to its degree. In other words, the more a node is connected to others, the less relative importance those connections have. 
Most GNN layers available in Spektral require their own type of pre-processing in order to work correctly. You can find all necessary tools for pre-processing `A` in `spektral.utils`.

In our example, the pre-processing required by GCN is: 

```python
from spektral import utils
A = utils.localpooling_filter(A).astype('f4')
```

And that's all! 
What's left now for us is to compile and train our model: 

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()
```

Note that we used the `weighted_metrics` argument instead of the usual `metrics`. This is due to the particular semi-supervised problem that we are dealing with, and has to do with the boolean masks that we loaded earlier (more on that later).

We can now train the model using the native `fit()` method of Keras, no modifications needed:

```python
# Train model
validation_data = ([X, A], y, val_mask)
model.fit([X, A],
          y,
          sample_weight=train_mask,
          epochs=100,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False)  # Shuffling data means shuffling the whole graph
``` 

There are a couple of things to note here.

We have set `batch_size=N` and `shuffle=False`. This is because, since we are considering our nodes as samples in a dataset, the default behaviour of Keras will be to split our graph into batches of 32, and shuffle the batches at each epoch. 
However, shuffling the adjacency matrix along one axis and not the other means that row `i` will represent a different node than column `i`. 
At the same time, if we split the graph in batches, we can easily end up in a situation where we need to use a node attribute that is not part of the batch. The only solution is to take all the node features at the same time, hence `batch_size=N`.

Finally, we used `train_mask` and `val_mask` as `sample_weight`.   
This means that, during training, the training nodes will have a weight of 1 and the validation nodes will have a weight of 0. Then, in validation, we will set the training nodes to have a weight of 0 and the validation nodes to have a weight of 1. 

This is all that we need to do to differentiate between training and test data. See how the model takes as input the full `X`, `A`, and `y` for both training and valdation? The only thing that changes is the mask. This is also why we used the `weighted_metrics` keyword when compiling the model, so that our accuracy is calculated only on the correct nodes at each phase. 

## Evaluating the model

Once again, evaluation is done in vanilla Keras. We just have to keep in mind the same considerations about batching that we did for training (note that in `model.evaluate()`, by default, `shuffle=False`): 

```python
# Evaluate model
eval_results = model.evaluate([X, A],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
```

Done! Our model has been defined, trained, and evaluated.

## Go create!

You are now ready to use Spektral to create your own models. 

If you want to build a GNN for a specific task, chances are that the things you need to define the model and pre-process the data are already part of Spektral. Check the [examples](https://github.com/danielegrattarola/spektral/tree/master/examples) for some ideas and practical tips.

Remember to read the [data representation section](https://danielegrattarola.github.io/spektral/data/) to learn about how GNNs can be used to solve different machine learning problems on graphs. 

Make sure to check the documentation, and leave a comment [on Github](https://github.com/danielegrattarola/spektral) if you have a feature that you want to see implemented. 
