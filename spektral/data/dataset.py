import copy
import os.path as osp
import warnings

import numpy as np
import tensorflow as tf

from spektral.data.graph import Graph
from spektral.data.utils import get_spec
from spektral.datasets.utils import DATASET_FOLDER


class Dataset:
    """
    A container for Graph objects. This class can be extended to represent a
    graph dataset.

    To create a `Dataset`, you must implement the `Dataset.read()` method, which
    must return a list of `spektral.data.Graph` objects:

    ```py
    class MyDataset(Dataset):
        def read(self):
            return [Graph(x=x, adj=adj, y=y) for x, adj, y in some_magic_list]
    ```

    The `download()` method is automatically called if the path returned by
    `Dataset.path` does not exists (default `~/.spektral/datasets/ClassName/`).

    In this case, `download()` will be called before `read()`.

    Datasets should generally behave like Numpy arrays for any operation that
    uses simple 1D indexing:

    ```py
    >>> dataset[0]
    Graph(...)

    >>> dataset[[1, 2, 3]]
    Dataset(n_graphs=3)

    >>> dataset[1:10]
    Dataset(n_graphs=9)

    >>> np.random.shuffle(dataset)  # shuffle in-place

    >>> for graph in dataset[:3]:
    >>>     print(graph)
    Graph(...)
    Graph(...)
    Graph(...)
    ```

    Datasets have the following properties that are automatically computed:

        - `n_nodes`: the number of nodes in the dataset (always None, except
        in single and mixed mode datasets);
        - `n_node_features`: the size of the node features (assumed to be equal
        for all graphs);
        - `n_edge_features`: the size of the edge features (assumed to be equal
        for all graphs);
        - `n_labels`: the size of the labels (assumed to be equal for all
        graphs); this is computed as `y.shape[-1]`.

    Any additional `kwargs` passed to the constructor will be automatically
    assigned as instance attributes of the dataset.

    Datasets also offer three main manipulation functions to apply callables to
    their graphs:

    - `apply(transform)`: replaces each graph with the output of `transform(graph)`.
    See `spektral.transforms` for some ready-to-use transforms.<br>
    Example: `apply(spektral.transforms.NormalizeAdj())` normalizes the
    adjacency matrix of each graph in the dataset.
    - `map(transform, reduce=None)`: returns a list containing the output
    of `transform(graph)` for each graph. If `reduce` is a `callable`, then
    returns `reduce(output_list)`.<br>
    Example: `map(lambda: g.n_nodes, reduce=np.mean)` will return the
    average number of nodes in the dataset.
    - `filter(function)`: removes from the dataset any graph for which
    `function(graph) is False`.<br>
    Example: `filter(lambda: g.n_nodes < 100)` removes from the dataset all
    graphs bigger than 100 nodes.

    Datasets in mixed mode (one adjacency matrix, many instances of node features)
    are expected to have a particular structure.
    The graphs returned by `read()` should not have an adjacency matrix,
    which should be instead stored as a singleton in the dataset's `a` attribute.
    For example:

    ```py
    class MyMixedModeDataset(Dataset):
        def read(self):
            self.a = compute_adjacency_matrix()
            return [Graph(x=x, y=y) for x, y in some_magic_list]
    ```

    Have a look at the `spektral.datasets` module for examples of popular
    datasets already implemented.

    **Arguments**

    - `transforms`: a callable or list of callables that are automatically
    applied to the graphs after loading the dataset.
    """

    def __init__(self, transforms=None, **kwargs):
        self.a = None  # Used for mixed-mode datasets
        # Read extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Download data
        if not osp.exists(self.path):
            self.download()

        # Read graphs
        self.graphs = self.read()
        if self.a is None and self.__len__() > 0 and "a" not in self.graphs[0]:
            warnings.warn(
                "The graphs in this dataset have no adjacency matrix. "
                "Is this intentional?"
            )

        # Apply transforms
        if transforms is not None:
            if not isinstance(transforms, (list, tuple)) and callable(transforms):
                transforms = [transforms]
            elif not all([callable(t) for t in transforms]):
                raise ValueError(
                    "`transforms` must be a callable or list of " "callables"
                )
            else:
                pass
            for t in transforms:
                self.apply(t)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass

    def apply(self, transform):
        if not callable(transform):
            raise ValueError("`transform` must be callable")

        for i in range(len(self.graphs)):
            self.graphs[i] = transform(self.graphs[i])

    def map(self, transform, reduce=None):
        if not callable(transform):
            raise ValueError("`transform` must be callable")
        if reduce is not None and not callable(reduce):
            raise ValueError("`reduce` must be callable")

        out = [transform(g) for g in self.graphs]
        return reduce(out) if reduce is not None else out

    def filter(self, function):
        if not callable(function):
            raise ValueError("`function` must be callable")
        self.graphs = [g for g in self.graphs if function(g)]

    def __getitem__(self, key):
        if not (
            np.issubdtype(type(key), np.integer)
            or isinstance(key, (slice, list, tuple, np.ndarray))
        ):
            raise ValueError("Unsupported key type: {}".format(type(key)))
        if np.issubdtype(type(key), np.integer):
            return self.graphs[int(key)]
        else:
            dataset = copy.copy(self)
            if isinstance(key, slice):
                dataset.graphs = self.graphs[key]
            else:
                dataset.graphs = [self.graphs[i] for i in key]
            return dataset

    def __setitem__(self, key, value):
        is_iterable = isinstance(value, (list, tuple))
        if not isinstance(value, (Graph, list, tuple)):
            raise ValueError(
                "Datasets can only be assigned Graphs or " "sequences of Graphs"
            )
        if is_iterable and not all([isinstance(v, Graph) for v in value]):
            raise ValueError("Assigned sequence must contain only Graphs")
        if is_iterable and isinstance(key, int):
            raise ValueError("Cannot assign multiple Graphs to one location")
        if not is_iterable and isinstance(key, (slice, list, tuple)):
            raise ValueError("Cannot assign one Graph to multiple locations")
        if not (isinstance(key, (int, slice, list, tuple))):
            raise ValueError("Unsupported key type: {}".format(type(key)))

        if isinstance(key, int):
            self.graphs[key] = value
        else:
            if isinstance(key, slice):
                self.graphs[key] = value
            else:
                for i, k in enumerate(key):
                    self.graphs[k] = value[i]

    def __len__(self):
        return len(self.graphs)

    def __repr__(self):
        return "{}(n_graphs={})".format(self.__class__.__name__, self.n_graphs)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, self.__class__.__name__)

    @property
    def n_graphs(self):
        return self.__len__()

    @property
    def n_nodes(self):
        if len(self.graphs) == 1:
            return self.graphs[0].n_nodes
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_node_features(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_node_features
        else:
            return None

    @property
    def n_edge_features(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_edge_features
        else:
            return None

    @property
    def n_labels(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_labels
        else:
            return None

    @property
    def signature(self):
        """
        This property computes the signature of the dataset, which can be
        passed to `spektral.data.utils.to_tf_signature(signature)` to compute
        the TensorFlow signature. You can safely ignore this property unless
        you are creating a custom `Loader`.

        A signature consist of the TensorFlow TypeSpec, shape, and dtype of
        all characteristic matrices of the graphs in the Dataset. This is
        returned as a dictionary of dictionaries, with keys `x`, `a`, `e`, and
        `y` for the four main data matrices.

        Each sub-dictionary will have keys `spec`, `shape` and `dtype`.
        """
        if len(self.graphs) == 0:
            return None
        signature = {}
        graph = self.graphs[0]  # This is always non-empty
        if graph.x is not None:
            signature["x"] = dict()
            signature["x"]["spec"] = get_spec(graph.x)
            signature["x"]["shape"] = (None, self.n_node_features)
            signature["x"]["dtype"] = tf.as_dtype(graph.x.dtype)
        if graph.a is not None:
            signature["a"] = dict()
            signature["a"]["spec"] = get_spec(graph.a)
            signature["a"]["shape"] = (None, None)
            signature["a"]["dtype"] = tf.as_dtype(graph.a.dtype)
        if graph.e is not None:
            signature["e"] = dict()
            signature["e"]["spec"] = get_spec(graph.e)
            signature["e"]["shape"] = (None, self.n_edge_features)
            signature["e"]["dtype"] = tf.as_dtype(graph.e.dtype)
        if graph.y is not None:
            signature["y"] = dict()
            signature["y"]["spec"] = get_spec(graph.y)
            signature["y"]["shape"] = (self.n_labels,)
            signature["y"]["dtype"] = tf.as_dtype(np.array(graph.y).dtype)
        return signature
