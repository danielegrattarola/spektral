### load_data


```python
spektral.datasets.mnist.load_data()
```



Loads the MNIST dataset and the associated grid.
This code is largely taken from [Michaël Defferrard's Github](https://github.com/mdeff/cnn_graph/blob/master/nips2016/mnist.ipynb).


**Return**  

- X_train, y_train: training node features and labels;
- X_val, y_val: validation node features and labels;
- X_test, y_test: test node features and labels;
- A: adjacency matrix of the grid;
