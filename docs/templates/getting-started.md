# Getting started

Spektral is designed according to the Keras API principles, in order to make things extremely simple for beginners, while maintaining flexibility for experts and researchers.  

The most important features of Spektral are the `layers.convolutional` and `layers.pooling` modules, which offer a number of popular layers to start building graph neural networks (GNNs) right away.     
Because Spektral is designed as an extension of Keras, you can plug any Spektral layer into an existing Keras `Model` without modifications. 

A good starting point is the collection of examples which can be found [on Github](https://github.com/danielegrattarola/spektral/tree/master/examples), and it is also a good idea to read the section on [how to represent graphs](https://danielegrattarola.github.io/spektral/data/) before starting this tutorial. 


## Node classification on citation networks

In this example, we will build a simple [Graph Convolutional Network for semi-supervised node classification](https://arxiv.org/abs/1609.02907).

This is a simple but challenging task that has caused GNN's recent rise to popularity, and consists in classifying documents in a **citation network**.   
In this network, each node represents a document, and node attributes are bag-of-words binary features. 
A link between two nodes means that one of the two documents cites the other.   
Finally, each node has a class label.

This is a **transductive** learning setting, where we observe all of the nodes and edges at training time, but only a fraction of the labels. The goal is to learn to predict the missing labels.

The `datasets.citation` module of Spektral allows you to download and load three popular citation datasets (Cora, Citeseer and Pubmed) in one line of code. For instance, loading the Cora dataset is as simple as: 

```python
from spektral.datasets import citation
data = citation.load_data('cora')
A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = data

N = A.shape[0]
F = X.shape[-1]
n_classes = y_train.shape[-1]
```

This will load the network's adjacency matrix (`A`) in a Scipy sparse format, the node features (`X`), and the pre-split training, validation, and test labels (`y_train, y_val, y_test`). The loader will also return some boolean masks to know which nodes belong to which set (`train_mask, val_mask, test_mask`).

We also saved a couple of values that will be useful later: the number of nodes in the graph (`N`), the size of the node attributes (`F`), and the number of classes in the labels (`n_classes`).


## Creating a GNN

To create a GCN similar to the one in the paper, we will use the `GraphConv` layer and the functional API of Keras:

```python
from spektral.layers import GraphConv
from keras.models import Model
from keras.layers import Input, Dropout
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

Note how we used the same familiar API of Keras for creating the GCN layers, as well as the standard `Dropout` layer to regularize our model. If we wanted, we could choose our own regularizers and initializers for the weights of `GraphConv` as well.

An important thing to keep in mind is that in this **single mode** (see the [data representation section](https://danielegrattarola.github.io/spektral/data/)), there is no batch dimension. The "elements" of our dataset are, in a sense, the node themselves. This is why we omitted the first dimension of `X` and `A` in the `Input` layers (`shape=(F, )` instead of `(N, F)`, and `shape=(N, )` instead of `(N, N)`). 

This will become clearer later. 

## Training the GNN

Before training the model, we have to do a simple pre-processing of the adjacency matrix, in order to scale the weight (i.e., the importance) of a node's connections according to the node's degree. In other words, the more a node is connected, the less relative importance those connections have (plus some other minor considerations that you can find in the [original GCN paper](https://arxiv.org/abs/1609.02907)).  

This is simply achieved by doing:

```python
from spektral import utils
A = utils.localpooling_filter(A)
```

which will give us a normalized adjacency matrix that we can use to train the GCN. 

What's left now for us is to compile and train our model: 

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()
```

Note that we used the `weighted_metrics` argument instead of the usual `metrics`. This is due to the particular semi-supervised problem that we are dealing with, and has to do with the boolean masks that we loaded earlier (more on that later).

We are now ready to train the model:

```python
# Train model
validation_data = ([X, A], y_val, val_mask)
model.fit([X, A],
          y_train,
          sample_weight=train_mask,
          epochs=100,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False)  # Shuffling data means shuffling the whole graph
``` 

There are a couple of things to note here.

First, we trained our model using the native `fit()` method of Keras. No modifications needed.

Second, we have set `batch_size=N` and `shuffle=False`. This is because we are working in **single mode**, meaning that Keras will interpret the first dimension of our adjacency and node attributes matrices as the "batch" dimension.    
If left to its own devices, Keras will automatically try to split our graph into batches of 32, and shuffle the batches at each epoch. For us, that means that the graph would get mixed and cut beyond repair, and the model would not be able to learn. This is why we tell Keras to use a batch size of `N` (the whole graph) and to not shuffle the nodes between epochs.  
This would not be necessary if we were working in **batch mode** instead, with many different graphs in our dataset. 

Finally, we used `train_mask` and `val_mask` as `sample_weight`.   
This results in the training nodes being assigned a weight of 1 during training, and the nodes outside the training set being assigned a weight of 0. The same holds for the validation and test sets.    
This is all that we need to do to differentiate between training and test data. See how the model takes as input the full `X` and `A` for both phases? The only thing that changes is the mask and targets. This is also why we used the `weighted_metrics` flag when compiling the model. 

## Evaluating the model

Again, this is done in vanilla Keras. We just have to keep in mind the same considerations about batching that we did for training (`shuffle` is `False` by default in `evaluate()`): 

```python
# Evaluate model
eval_results = model.evaluate([X, A],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
```

Done! Our model has been defined, trained, and evaluated.

## Go create!

If you made it to this point, you are ready to use Spektral to create your own models. 

If you want to build a GNN for a specific task, chances are that the things you need to define the model and pre-process the data are already part of Spektral. Check the [examples](https://github.com/danielegrattarola/spektral/tree/master/examples) for some ideas and practical tips.

Remember to read the [data representation section](https://danielegrattarola.github.io/spektral/data/) to learn about how GNNs can be used to solve different problems. 

Make sure to check the documentation, and leave a comment [on Github](https://github.com/danielegrattarola/spektral) if you have a feature that you want to see implemented. 
