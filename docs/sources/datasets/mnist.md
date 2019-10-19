### load_data


```python
spektral.datasets.mnist.load_data(k=8, noise_level=0.0)
```



Loads the MNIST dataset and a K-NN grid.
This code is largely taken from [Michaël Defferrard's Github](https://github.com/mdeff/cnn_graph/blob/master/nips2016/mnist.ipynb).


**Arguments**  

- ` k`: int, number of neighbours for each node;

- ` noise_level`: fraction of edges to flip (from 0 to 1 and vice versa);


**Return**  

- X_train, y_train: training node features and labels;
- X_val, y_val: validation node features and labels;
- X_test, y_test: test node features and labels;
- A: adjacency matrix of the grid;
