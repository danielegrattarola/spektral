# Creating a Custom Dataset

The `Dataset` class is a new feature of Spektral 1.0 that standardizes how graph datasets are represented in Spektral. 

In this tutorial, we'll go over a simple example to create a custom dataset with your data. 

This is also useful if you want to share you datasets publicly or include them as part of Spektral. 

## Essential information

You create a dataset by subclassing the `spektral.data.Dataset` class. 

The core of datasets is the `read()` method. This is called at every instantiation of the dataset and must return a list of `spektral.data.Graph`.
It doesn't matter if you read the data from a file or create it on the fly, this is where the dataset is loaded in memory. 

All datasets have a `path` property that represents the directory in which the data is stored. This defaults to `~/spektral/datasets/[ClassName]`.
You can ignore it if you want.<br>
However, each time you instantiate a Dataset it will check whether `path` exists. If it doesn't, the `download()` method will be called.

You can use `download()` to define any additional operations that are needed to save your raw data to disk. The method will be called **before** `read()`.

Both `read()` and `download()` are called by the dataset's `__init__()` method. If you need to override the initialization of your dataset, make sure to call `super().__init__()` somewhere in your implementation (usually as the last line).

## Example

This is a simple example that shows how to create a custom dataset with five random graphs. We pretend that the data comes from an online source so that we can show how to use `download()`. 

We start by overriding the `__init__()` method to store some custom parameters of the dataset: 

```py
class MyDataset(Dataset):
    """
    A dataset of five random graphs.
    """
    def __init__(self, nodes, feats, **kwargs):
        self.nodes = nodes
        self.feats = feats

        super().__init__(**kwargs)
```

Remember to call `super().__init__(**kwargs)` as the last line.

Then, we simulate downloading the data from the web. Since this method gets called if `path` does not exist on the system, it makes sense to create the corresponding directory now:

```py
def download(self):
    data = ...  # Download from somewhere

    # Create the directory
    os.mkdir(self.path)

    # Write the data to file
    for i in range(5):
        x = rand(self.nodes, self.feats)
        a = randint(0, 2, (self.nodes, self.nodes))
        y = randint(0, 2)

        filename = os.path.join(self.path, f'graph_{i}')
        np.savez(filename, x=x, a=a, y=y)
```

Finally, we implement the `read()` method to return a list of `Graph` objects:

```py
def read(self):
    # We must return a list of Graph objects
    output = []
    
    for i in range(5):
        data = np.load(os.path.join(self.path, f'graph_{i}.npz'))
        output.append(
            Graph(x=data['x'], a=data['a'], y=data['y'])
        )

    return output
```

We can now instantiate our dataset, which will "download" our data and read it into memory: 

```
>>> dataset = MyDataset(3, 2)
>>> dataset
MyDataset(n_graphs=5)
```

We can see that our graphs were saved to file: 

```sh
$ ls ~/.spektral/datasets/MyDataset/
graph_0.npz  graph_1.npz  graph_2.npz  graph_3.npz  graph_4.npz
```

so the next time we create `MyDataset` it will read from the files we have saved. 

---

You can now use your custom dataset however you like. [Loaders](/loaders) will work, as well as [transforms](/transforms) and all other features described in the [documentation](/data/#dataset).

Remember that, if you want, you're free to store your data as you prefer. Datasets in Spektral are just there to simplify your workflow, but the library is still designed according to Keras' principle of not getting in your way. If you want to manipulate your data differently, your GNNs will still work. 

You can also see [this script](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/custom_dataset.py) for another example on how to create and use a custom dataset.
