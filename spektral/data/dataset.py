import copy
import os.path as osp

import numpy as np
import tensorflow as tf

from spektral.data import Graph
from spektral.data.utils import get_spec
from spektral.datasets.utils import DATASET_FOLDER


class Dataset:
    """
    A container for Graph objects. This class can be extended to represent a
    graph dataset.

    To extend this class, you must implement the `Dataset.read()` method, which
    must return a list of `spektral.data.Graph` objects, e.g.,

    ```
    class MyDataset(Dataset):
        def read(self):
            return [
                Graph(x=np.random.rand(n, 2),
                      adj=np.random.randint(0, 2, (n, n)),
                      y=np.array([0., 1.]))
                for n in range(size)
            ]
    ```

    Datasets can be sliced (`dataset[start:stop]`), shuffled
    (`np.random.shuffle(dataset)`), and iterated (`for graph in dataset: ...`).

    The size of the node features, edge features and targets is shared by all
    graphs in a dataset and can be accessed respectively with:

    ```
    >>> dataset.F
    >>> dataset.s_1
    >>> dataset.n_out
    ```

    The general shape, dtype, and `tf.TypeSpec` of the matrices composing the
    graphs is stored in `dataset.signature`. This can be useful when
    implementing a custom Loader for your dataset.
    """
    def __init__(self, transforms=None, **kwargs):
        if not osp.exists(self.path):
            self.download()
        self.graphs = self.read()
        # Make sure that we always have at least one graph
        if len(self.graphs) == 0:
            raise ValueError('Datasets cannot be empty')

        if len(self.graphs) == 1 or len(set([g.N for g in self.graphs])) == 1:
            self.N = self.graphs[0].N
        else:
            self.N = None
        self.F = None
        self.S = None
        self.n_out = None
        self.signature = self._signature()

        # Read extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Apply transforms
        if transforms is not None:
            if not isinstance(transforms, (list, tuple)) and callable(transforms):
                transforms = [transforms]
            elif not all([callable(t) for t in transforms]):
                raise ValueError('transforms must be a list of callables or '
                                 'a callable.')
            else:
                pass
            for t in transforms:
                self.apply(t)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, self.__class__.__name__)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass

    def apply(self, transform):
        for i in range(len(self.graphs)):
            self.graphs[i] = transform(self.graphs[i])

    def map(self, transform, reduce=None):
        out = [transform(g) for g in self.graphs]
        if reduce is not None and callable(reduce):
            return reduce(out)
        else:
            return out

    def _signature(self):
        signature = {}
        graph = self.graphs[0]  # This is always non-empty
        if graph.x is not None:
            signature['x'] = dict()
            signature['x']['spec'] = get_spec(graph.x)
            signature['x']['shape'] = (None, graph.F)
            signature['x']['dtype'] = tf.as_dtype(graph.x.dtype)
            self.F = graph.F
        if graph.adj is not None:
            signature['a'] = dict()
            signature['a']['spec'] = get_spec(graph.adj)
            signature['a']['shape'] = (None, None)
            signature['a']['dtype'] = tf.as_dtype(graph.adj.dtype)
        if graph.edge_attr is not None:
            signature['e'] = dict()
            signature['e']['spec'] = get_spec(graph.edge_attr)
            signature['e']['shape'] = (None, graph.s_1)
            signature['e']['dtype'] = tf.as_dtype(graph.edge_attr.dtype)
            self.S = graph.s_1
        if graph.y is not None:
            signature['y'] = dict()
            signature['y']['spec'] = get_spec(graph.y)
            signature['y']['shape'] = (graph.y.shape[-1], )
            signature['y']['dtype'] = tf.as_dtype(graph.y.dtype)
            self.n_out = graph.y.shape[-1]

        return signature

    def __getitem__(self, key):
        if not (np.issubdtype(type(key), np.integer) or
                isinstance(key, (slice, list, tuple, np.ndarray))):
            raise ValueError('Unsupported key type: {}'.format(type(key)))
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
            raise ValueError('Datasets can only be assigned Graphs or '
                             'sequences of Graphs')
        if is_iterable and not all([isinstance(v, Graph) for v in value]):
            raise ValueError('Assigned sequence must contain only Graphs')
        if is_iterable and isinstance(key, int):
            raise ValueError('Cannot assign multiple Graphs to one location')
        if not is_iterable and isinstance(key, (slice, list, tuple)):
            raise ValueError('Cannot assign one Graph to multiple locations')
        if not (isinstance(key, (int, slice, list, tuple))):
            raise ValueError('Unsupported key type: {}'.format(type(key)))

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
        return 'Dataset(len={}, signature="{}")'\
            .format(self.__len__(), ', '.join(self.signature.keys()))