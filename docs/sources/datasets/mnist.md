### load_data


```python
spektral.datasets.mnist.load_data(k=8, noise_level=0.0)
```



Loads the MNIST dataset and a K-NN graph to perform graph signal
classification, as described by [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).
The K-NN graph is statically determined from a regular grid of pixels using
the 2d coordinates.

The node features of each graph are the MNIST digits vectorized and rescaled
to [0, 1].
Two nodes are connected if they are neighbours according to the K-NN graph.
Labels are the MNIST class associated to each sample.


**Arguments**  

- ` k`: int, number of neighbours for each node;

- ` noise_level`: fraction of edges to flip (from 0 to 1 and vice versa);


**Return**  

- X_train, y_train: training node features and labels;
- X_val, y_val: validation node features and labels;
- X_test, y_test: test node features and labels;
- A: adjacency matrix of the grid;
