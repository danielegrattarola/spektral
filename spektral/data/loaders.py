import numpy as np
import tensorflow as tf

from spektral.data.utils import (
    batch_generator,
    collate_labels_batch,
    collate_labels_disjoint,
    get_spec,
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_batch,
    to_disjoint,
    to_mixed,
    to_tf_signature,
)

version = tf.__version__.split(".")
major, minor = int(version[0]), int(version[1])
tf_loader_available = major >= 2 and minor >= 4


class Loader:
    """
    Parent class for data loaders. The role of a Loader is to iterate over a
    Dataset and yield batches of graphs to feed your Keras Models.

    This is achieved by having a generator object that produces lists of Graphs,
    which are then collated together and returned as Tensors.

    The core of a Loader is the `collate(batch)` method.
    This takes as input a list of `Graph` objects and returns a list of Tensors,
    np.arrays, or SparseTensors.

    For instance, if all graphs have the same number of nodes and size of the
    attributes, a simple collation function can be:

    ```python
    def collate(self, batch):
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch)]
        return x, a
    ```

    The `load()` method of a Loader returns an object that can be passed to a Keras
    model when using the `fit`, `predict` and `evaluate` functions.
    You can use it as follows:

    ```python
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)
    ```

    The `steps_per_epoch` property represents the number of batches that are in
    an epoch, and is a required keyword when calling `fit`, `predict` or `evaluate`
    with a Loader.

    If you are using a custom training function, you can specify the input signature
    of your batches with the tf.TypeSpec system to avoid unnecessary re-tracings.
    The signature is computed automatically by calling `loader.tf_signature()`.

    For example, a simple training step can be written as:

    ```python
    @tf.function(input_signature=loader.tf_signature())  # Specify signature here
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ```

    We can then train our model in a loop as follows:

    ```python
    for batch in loader:
        train_step(*batch)
    ```

    **Arguments**

    - `dataset`: a `spektral.data.Dataset` object;
    - `batch_size`: int, size of the mini-batches;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the dataset at the start of each epoch.
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
        """
        Returns lists (batches) of `Graph` objects.
        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        """
        Converts a list of graph objects to Tensors or np.arrays representing the batch.
        :param batch: a list of `Graph` objects.
        """
        raise NotImplementedError

    def load(self):
        """
        Returns an object that can be passed to a Keras model when using the `fit`,
        `predict` and `evaluate` functions.
        By default, returns the Loader itself, which is a generator.
        """
        return self

    def tf_signature(self):
        """
        Returns the signature of the collated batches using the tf.TypeSpec system.
        By default, the signature is that of the dataset (`dataset.signature`):

            - Adjacency matrix has shape `[n_nodes, n_nodes]`
            - Node features have shape `[n_nodes, n_node_features]`
            - Edge features have shape `[n_edges, n_node_features]`
            - Targets have shape `[..., n_labels]`
        """
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def pack(self, batch):
        """
        Given a batch of graphs, groups their attributes into separate lists and packs
        them in a dictionary.

        For instance, if a batch has three graphs g1, g2 and g3 with node
        features (x1, x2, x3) and adjacency matrices (a1, a2, a3), this method
        will return a dictionary:

        ```python
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```

        :param batch: a list of `Graph` objects.
        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]
        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        """
        :return: the number of batches of size `self.batch_size` in the dataset (i.e.,
        how many batches are in an epoch).
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))


class SingleLoader(Loader):
    """
    A Loader for [single mode](https://graphneural.network/data-modes/#single-mode).

    This loader produces Tensors representing a single graph. As such, it can
    only be used with Datasets of length 1 and the `batch_size` cannot be set.

    The loader supports sample weights through the `sample_weights` argument.
    If given, then each batch will be a tuple `(inputs, labels, sample_weights)`.

    **Arguments**

    - `dataset`: a `spektral.data.Dataset` object with only one graph;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch;
    - `sample_weights`: Numpy array, will be appended to the output automatically.

    **Output**

    Returns a tuple `(inputs, labels)` or `(inputs, labels, sample_weights)`.

    `inputs` is a tuple containing the data matrices of the graph, only if they
    are not `None`:

    - `x`: same as `dataset[0].x`;
    - `a`: same as `dataset[0].a` (scipy sparse matrices are converted to
    SparseTensors);
    - `e`: same as `dataset[0].e`;

    `labels` is the same as `dataset[0].y`.

    `sample_weights` is the same array passed when creating the loader.
    """

    def __init__(self, dataset, epochs=None, sample_weights=None):
        if len(dataset) != 1:
            raise ValueError(
                "SingleLoader can only be used with Datasets that"
                "have a single graph."
            )
        self.sample_weights = sample_weights
        super().__init__(dataset, batch_size=1, epochs=epochs, shuffle=False)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=True)

        output = to_disjoint(**packed)
        output = output[:-1]  # Discard batch index
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        output = (output,)
        if y is not None:
            output += (y,)
        if self.sample_weights is not None:
            output += (self.sample_weights,)

        if len(output) == 1:
            output = output[0]  # Again, in case there are no targets and no SW

        return output

    def load(self):
        output = self.collate(self.dataset)
        return tf.data.Dataset.from_tensors(output).repeat(self.epochs)


class DisjointLoader(Loader):
    """
    A Loader for [disjoint mode](https://graphneural.network/data-modes/#disjoint-mode).

    This loader represents a batch of graphs via their disjoint union.

    The loader automatically computes a batch index tensor, containing integer
    indices that map each node to its corresponding graph in the batch.

    The adjacency matrix os returned as a SparseTensor, regardless of the input.

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are stacked along an additional dimension.
    If `node_level=True`, then the labels are stacked vertically.

    **Note:** TensorFlow 2.4 or above is required to use this Loader's `load()`
    method in a Keras training loop.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `node_level`: bool, if `True` stack the labels vertically for node-level
    prediction;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[n_nodes, n_node_features]`;
    - `a`: adjacency matrices of shape `[n_nodes, n_nodes]`;
    - `e`: edge attributes of shape `[n_edges, n_edge_features]`;
    - `i`: batch index of shape `[n_nodes]`.

    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[n_nodes, n_labels]` otherwise.

    """

    def __init__(
        self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True
    ):
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def load(self):
        if not tf_loader_available:
            raise RuntimeError(
                "Calling DisjointLoader.load() requires " "TensorFlow 2.4 or greater."
            )
        return tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature()
        )

    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [n_nodes, n_node_features]
        Edge features have shape [n_edges, n_edge_features]
        Targets have shape [*, n_labels]
        """
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        return to_tf_signature(signature)


class BatchLoader(Loader):
    """
    A Loader for [batch mode](https://graphneural.network/data-modes/#batch-mode).

    This loader returns batches of graphs stacked along an extra dimension,
    with all "node" dimensions padded to be equal among all graphs.

    If `n_max` is the number of nodes of the biggest graph in the batch, then
    the padding consist of adding zeros to the node features, adjacency matrix,
    and edge attributes of each graph so that they have shapes
    `[n_max, n_node_features]`, `[n_max, n_max]`, and
    `[n_max, n_max, n_edge_features]` respectively.

    The zero-padding is done batch-wise, which saves up memory at the cost of
    more computation. If latency is an issue but memory isn't, or if the
    dataset has graphs with a similar number of nodes, you can use
    the `PackedBatchLoader` that zero-pads all the dataset once and then
    iterates over it.

    Note that the adjacency matrix and edge attributes are returned as dense
    arrays.

    if `mask=True`, node attributes will be extended with a binary mask that indicates
    valid nodes (the last feature of each node will be 1 if the node was originally in
    the graph and 0 if it is a fake node added by zero-padding).

    Use this flag in conjunction with layers.base.GraphMasking to start the propagation
    of masks in a model (necessary for node-level prediction and models that use a
    dense pooling layer like DiffPool or MinCutPool).

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are returned as an array of shape `[batch, n_labels]`.
    If `node_level=True`, then the labels are padded along the node dimension and are
    returned as an array of shape `[batch, n_max, n_labels]`.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `mask`: bool, whether to add a mask to the node features;
    - `batch_size`: int, size of the mini-batches;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch;
    - `node_level`: bool, if `True` pad the labels along the node dimension;

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.

    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[batch, n_max, n_labels]` otherwise.
    """

    def __init__(
        self,
        dataset,
        mask=False,
        batch_size=1,
        epochs=None,
        shuffle=True,
        node_level=False,
    ):
        self.mask = mask
        self.node_level = node_level
        self.signature = dataset.signature
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_batch(y, node_level=self.node_level)

        output = to_batch(**packed, mask=self.mask)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Labels have shape [batch, n_labels]
        """
        signature = self.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature and self.mask:
            # In case we have a mask, the mask is concatenated to the features
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])
        if "y" in signature and self.node_level:
            # Node labels have an extra None dimension
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])

        return to_tf_signature(signature)


class PackedBatchLoader(BatchLoader):
    """
    A `BatchLoader` that zero-pads the graphs before iterating over the dataset.
    This means that `n_max` is computed over the whole dataset and not just
    a single batch.

    While using more memory than `BatchLoader`, this loader should reduce the
    computational overhead of padding each batch independently.

    Use this loader if:

    - memory usage isn't an issue and you want to produce the batches as fast
    as possible;
    - the graphs in the dataset have similar sizes and there are no outliers in
    the dataset (i.e., anomalous graphs with many more nodes than the dataset
    average).

    **Arguments**

    - `dataset`: a graph Dataset;
    - `mask`: bool, whether to add a mask to the node features;
    - `batch_size`: int, size of the mini-batches;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch;
    - `node_level`: bool, if `True` pad the labels along the node dimension;

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.

    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[batch, n_max, n_labels]` otherwise.
    """

    def __init__(
        self,
        dataset,
        mask=False,
        batch_size=1,
        epochs=None,
        shuffle=True,
        node_level=False,
    ):
        super().__init__(
            dataset,
            mask=mask,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            node_level=node_level,
        )

        # Drop the Dataset container and work on packed tensors directly
        packed = self.pack(self.dataset)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_batch(y, node_level=self.node_level)

        self.dataset = to_batch(**packed, mask=mask)
        if y is not None:
            self.dataset += (y,)

        # Re-instantiate generator after packing dataset
        self._generator = self.generator()

    def collate(self, batch):
        if len(batch) == 2:
            # If there is only one input, i.e., batch = [x, y], we unpack it
            # like this because Keras does not support input lists with only
            # one tensor.
            return batch[0], batch[1]
        else:
            return batch[:-1], batch[-1]

    @property
    def steps_per_epoch(self):
        if len(self.dataset) > 0:
            return int(np.ceil(len(self.dataset[0]) / self.batch_size))


class MixedLoader(Loader):
    """
    A Loader for [mixed mode](https://graphneural.network/data-modes/#mixed-mode).

    This loader returns batches where the node and edge attributes are stacked
    along an extra dimension, but the adjacency matrix is shared by all graphs.

    The loader expects all node and edge features to have the same number of
    nodes and edges.
    The dataset is pre-packed like in a PackedBatchLoader.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `batch_size`: int, size of the mini-batches;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[batch, n_nodes, n_node_features]`;
    - `a`: adjacency matrix of shape `[n_nodes, n_nodes]`;
    - `e`: edge attributes of shape `[batch, n_edges, n_edge_features]`.

    `labels` have shape `[batch, ..., n_labels]`.

    """

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        assert dataset.a is not None, (
            "Dataset must be in mixed mode, with only "
            "one adjacency matrix stored in the "
            "dataset's `a` attribute.\n"
            "If your dataset does not have an "
            "adjacency matrix, you can use a "
            "BatchLoader or PackedBatchLoader instead."
        )
        assert "a" not in dataset.signature, (
            "Datasets in mixed mode should not"
            "have the adjacency matrix stored"
            "in their Graph objects."
        )
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)
        packed["a"] = self.dataset.a

        y = packed.pop("y_list", None)
        if y is not None:
            y = np.array(y)

        output = to_mixed(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_edges, n_edge_features]
        Targets have shape [batch, ..., n_labels]
        """
        signature = self.dataset.signature
        for k in ["x", "e", "y"]:
            if k in signature:
                signature[k]["shape"] = prepend_none(signature[k]["shape"])

        signature["a"] = dict()
        signature["a"]["spec"] = get_spec(self.dataset.a)
        signature["a"]["shape"] = (None, None)
        signature["a"]["dtype"] = tf.as_dtype(self.dataset.a.dtype)

        return to_tf_signature(signature)
