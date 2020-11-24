import numpy as np
import tensorflow as tf
from scipy import sparse as sp

from spektral.data.utils import prepend_none, to_tf_signature, to_disjoint, to_batch, batch_generator
from spektral.layers.ops import sp_matrix_to_sp_tensor

version = tf.__version__.split('.')
major, minor = int(version[0]), int(version[1])
tf_loader_available = major > 2 and minor > 3


class Loader:
    """
    Parent class for data loaders. The role of a Loader is to iterate over a
    Dataset and yield batches of graphs to give as input to your Keras Models.
    This is achieved by having a generator object that produces lists of Graphs,
    which are then collated together and returned as Tensors (or objects that
    can be converted to Tensors, like Numpy arrays).

    The core of a Loader is the `collate(batch)` method.
    This takes as input a list of Graphs and returns a list of Tensors or
    SparseTensors.

    For instance, if all graphs have the same number of nodes and size of the
    attributes, a simple collation function can be:

    ```python
    def collate(self, batch):
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch)]
        return x, a
    ```

    Since all data matrices (node attributes, adjacency matrices, etc.)
    are usually collated together, the two list comprehensions of the example
    above can be computed all at once by using the private `_pack()` method
    of the Loader class:

    ```python
    def collate(self, batch):
        x, a = self._pack(batch)
        return np.array(x), np.array(a)
    ```

    Additionally, a Loader should implement two main methods that simplify its
    usage within the TensorFlow/Keras training pipeline:

    - `load()`: should return a `tf.data` dataset, a generator, or a
    `keras.utils.Sequence`. Its usage pattern should be as follows:

    `model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)`

    The `steps_per_epoch` property returns the number of batches
    (as specified by the `batch_size` argument) that are in an epoch and is
    automatically computed from the data.

    By default, `load()` will simply return a `tf.data.Dataset.from_generator`
    dataset obtained from the Loader itself (since Loaders are also Python
    generators).

    - `tf_signature()`: this method should return the Tensorflow signature of
    the batches computed by `collate(batch)`, using the `tf.TypeSpec` system.
    All Datasets have a `signature` property that can be used to compute the
    TensorFlow signature (which represents the shape, dtype and TypeSpec of a
    each data matrix in a generic graph) with the
    `spektral.data.utils.to_tf_signature(signature)` function.

    By default, `tf_signature()` will simply return the Dataset's signature
    converted to the TensorFlow format.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.
    """
    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        return batch_generator(self.dataset, batch_size=self.batch_size,
                               epochs=self.epochs, shuffle=self.shuffle)

    def collate(self, batch):
        raise NotImplementedError

    def load(self):
        return self

    def tf_signature(self):
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def _pack(self, batch):
        return [list(elem) for elem in zip(*[g.numpy() for g in batch])]

    @property
    def steps_per_epoch(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))


class SingleLoader(Loader):
    """
    A Loader for single mode.

    This loader produces Tensors representing one graph, with its attributes,
    edges and labels (usually for node-level prediction). As such, it can only
    be used with Datasets of length 1 and the `batch_size` cannot be set.

    The loader also supports sample weights through the `sample_weights`
    argument. If given, then each batch will be a tuple
    `(inputs, labels, sample_weights)`.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch;
    - `sample_weights`: if given, these will be appended to the output
    automatically.

    **Output**

    Returns a tuple `(inputs, labels)` or `(inputs, labels, sample_weights)`.

    `inputs` are a tuple containing the non-None data matrices of the graph:

    - `x`: same as `dataset[0].x`;
    - `a`: same as `dataset[0].a` (scipy sparse matrices are converted to
    SparseTensors);
    - `e`: same as `dataset[0].e`;

    `labels` is the same as `datsaset[0].y`.
    If available, `sample_weights` is the same object passed to the constructor.


    """
    def __init__(self, dataset, epochs=None, sample_weights=None):
        if len(dataset) != 1:
            raise ValueError('SingleLoader can only be used with Datasets that'
                             'have a single graph.')
        self.sample_weights = sample_weights
        super().__init__(dataset, batch_size=1, epochs=epochs, shuffle=False)

    def collate(self, batch):
        graph = batch[0]
        output = graph.numpy()
        output = [output[:-1], output[-1]]

        # Sparse matrices to SparseTensors
        for i in range(len(output)):
            if sp.issparse(output[i]):
                output[i] = sp_matrix_to_sp_tensor(output[i])

        if self.sample_weights is not None:
            output += [self.sample_weights]
        return tuple(output)

    def load(self):
        output = self.collate(self.dataset)
        return tf.data.Dataset.from_tensors(output).repeat(self.epochs)


class DisjointLoader(Loader):
    """
    A Loader for disjoint mode.

    This loader produces batches of graphs as their disjoint union, and supports
    labels both for graph-level and node-level learning.

    Because in disjoint mode we need a way to keep track of which nodes belong
    to which graph, the loader will also automatically compute a batch index
    tensor, containing integer indices that map each node to its corresponding
    graph in the batch.

    The adjacency matrix will always be returned as a SparseTensor, regardless
    of the input.
    Edge attributes will be returned as a sparse edge list of shape
    `(n_edges, n_edge_features)`.

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are stacked along an additional dimension (i.e., `(n_graphs, n_labels)`)
    If `node_level=True`, then the labels are stacked vertically (i.e.,
    `(n_nodes, n_labels)`).

    Note that TensorFlow 2.4 or above is required to use this Loader's `load()`
    method in a Keras training loop.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `node_level`: boolean (default `False`), whether to interpret labels as
    node-level instead of graph-level;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes stacked along the outermost dimension;
    - `a`: SparseTensor, the block-diagonal matrix obtained from the adjacency
    matrices of the batch;
    - `e`: edge attributes as edge list of shape `(n_edges, n_edge_features)`;

    If `node_level=False`, `labels` has shape `(n_graphs, n_labels)`;
    If `node_level=True`, then the labels are stacked vertically, i.e.,
    `(n_nodes, n_labels)`.

    """
    def __init__(self, dataset, node_level=False, batch_size=1, epochs=None,
                 shuffle=True):
        self.node_level = node_level
        super(DisjointLoader, self).__init__(dataset, batch_size=batch_size,
                                             epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self._pack(batch)
        if self.node_level:
            y = np.vstack(packed[-1])
        else:
            y = np.array(packed[-1])
        output = to_disjoint(*packed[:-1])

        # Sparse matrices to SparseTensors
        output = list(output)
        for i in range(len(output)):
            if sp.issparse(output[i]):
                output[i] = sp_matrix_to_sp_tensor(output[i])
        output = tuple(output)

        return output, y

    def load(self):
        if not tf_loader_available:
            raise RuntimeError('Calling DisjointLoader.load() requires '
                               'TensorFlow 2.4 or greater.')
        return tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature())

    def tf_signature(self):
        signature = self.dataset.signature
        if 'y' in signature:
            if not self.node_level:
                signature['y']['shape'] = prepend_none(signature['y']['shape'])
        if 'a' in signature:
            signature['a']['spec'] = tf.SparseTensorSpec

        signature['i'] = dict()
        signature['i']['spec'] = tf.TensorSpec
        signature['i']['shape'] = (None,)
        signature['i']['dtype'] = tf.as_dtype(tf.int64)

        return to_tf_signature(signature)


class BatchLoader(Loader):
    """
    A Loader for batch mode.

    This loader returns batches of graphs stacked along an extra dimension,
    with all "node" dimensions padded to be equal among all graphs.

    If `n_max` is the number of nodes of the biggest graph in the batch, then
    the padding consist of adding zeros to the node features, adjacency matrix,
    and edge attributes of each graph so that they have shapes
    `(n_max, n_node_features)`, `(n_max, n_max)`, and `(n_max, n_max, n_edge_features)` respectively.

    The zero-padding is done batch-wise, which saves up memory at the cost of
    more computation. If latency is an issue but memory isn't, or if the
    dataset has graphs with a similar number of nodes, you can use
    the `PackedBatchLoader` that first zero-pads all the dataset and then
    iterates over it.

    Note that the adjacency matrix and edge attributes are returned as dense
    arrays (mostly due to the lack of support for sparse tensor operations for
    rank >2).

    Only graph-level labels are supported with this loader (i.e., labels are not
    zero-padded because they are assumed to have no "node" dimensions).

    **Arguments**

    - `dataset`: a graph Dataset;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes, zero-padded and stacked along an extra dimension
    (shape `(n_graphs, n_max, n_node_features)`);
    - `a`: adjacency matrices (dense), zero-padded and stacked along an extra
    dimension (shape `(n_graphs, n_max, n_max)`);
    - `e`: edge attributes (dense), zero-padded and stacked along an extra
    dimension (shape `(n_graphs, n_max, n_max, n_edge_features)`).

    `labels` are also stacked along an extra dimension.

    """
    def collate(self, batch):
        packed = self._pack(batch)
        y = np.array(packed[-1])
        output = to_batch(*packed[:-1])
        if len(output) == 1:
            output = output[0]

        return output, y

    def tf_signature(self):
        signature = self.dataset.signature
        for k in signature:
            signature[k]['shape'] = prepend_none(signature[k]['shape'])
        if 'a' in signature:
            # Adjacency matrix in batch mode is dense
            signature['a']['spec'] = tf.TensorSpec
        if 'e' in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature['e']['shape'] = prepend_none(signature['e']['shape'])

        return to_tf_signature(signature)


class PackedBatchLoader(BatchLoader):
    """
    A `BatchLoader` that pre-pads the graphs before iterating over the dataset
    to create the batches.

    While using more memory than `BatchLoader`, this loader should reduce the
    computational overhead of padding each batch independently.

    Use this loader if:

    - memory usage isn't an issue and you want to compute the batches as fast
    as possible;
    - the graphs in the dataset have similar sizes and there are no outliers in
    the dataset (i.e., anomalous graphs with many more nodes than the dataset
    average).

    This loader is also useful for loading mixed-mode datsets, because it
    allows to create "standard" batches of node features with almost no overhead.
    """
    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
        # Drop the Dataset container and work on packed tensors directly
        self.dataset = self._pack(self.dataset)
        self.dataset = to_batch(*self.dataset[:-1]) + (np.array(self.dataset[-1]), )

        # Re-instantiate generator after packing dataset
        self._generator = self.generator()

    def collate(self, batch):
        if len(batch) == 2:
            return batch[0], batch[1]
        else:
            return batch[:-1], batch[-1]